import pandas as pd
from dotenv import load_dotenv
from src.database import get_sqldb_connection


load_dotenv(override=True)

TABLE_NAME = "retrieval_experiments"
SORT_COLUMNS = {
    "Newest": "timestamp",
    "MRR": "mrr",
    "nDCG": "ndcg",
    "Coverage": "coverage",
    "Retriever": "retriever",
    "Embedding model": "embedding_model",
}


def table_exists() -> bool:
    with get_sqldb_connection() as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
            (TABLE_NAME,),
        ).fetchone()
    return row is not None


def empty_experiments_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "id",
            "timestamp",
            "embedding_model",
            "retriever",
            "reranker",
            "prompt_version",
            "chunk_size",
            "chunk_overlap",
            "RETRIEVAL_K",
            "mrr",
            "ndcg",
            "coverage",
            "time_taken",
            "chunking_strategy"
        ]
    )


def load_experiments(
    search: str = "",
    sort_label: str = "Newest",
    sort_direction: str = "Descending",
    limit: int = 100,
) -> pd.DataFrame:
    if not table_exists():
        return empty_experiments_df()

    sort_column = SORT_COLUMNS.get(sort_label, "timestamp")
    direction = "ASC" if sort_direction == "Ascending" else "DESC"
    limit = max(1, min(int(limit or 100), 1000))

    where_clause = ""
    params: list[str | int] = []
    if search and search.strip():
        term = f"%{search.strip()}%"
        where_clause = """
            WHERE embedding_model LIKE ?
               OR retriever LIKE ?
               OR reranker LIKE ?
               OR prompt_version LIKE ?
        """
        params.extend([term, term, term, term])

    query = f"""
        SELECT
            id,
            timestamp,
            embedding_model,
            retriever,
            reranker,
            prompt_version,
            chunk_size,
            chunk_overlap,
            RETRIEVAL_K,
            mrr,
            ndcg,
            coverage,
            time_taken,
            chunking_strategy
        FROM {TABLE_NAME}
        {where_clause}
        ORDER BY {sort_column} {direction}, id {direction}
        LIMIT ?
    """
    params.append(limit)

    with get_sqldb_connection() as conn:
        df = pd.read_sql_query(query, conn, params=params)

    for column in ["mrr", "ndcg", "coverage"]:
        if column in df:
            df[column] = df[column].round(4)
    return df


def build_summary(df: pd.DataFrame) -> str:
    if df.empty:
        return f"""
        <div class="summary-grid">
            <div class="metric-card">
                <div class="metric-label">Runs Shown</div>
                <div class="metric-main">0</div>
            </div>
        </div>
        """

    avg_mrr = df["mrr"].mean()
    avg_ndcg = df["ndcg"].mean()
    avg_coverage = df["coverage"].mean()
    best_mrr = df["mrr"].max()

    return f"""
    <div class="summary-grid">
        <div class="metric-card">
            <div class="metric-label">Runs Shown</div>
            <div class="metric-main">{len(df)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Average MRR</div>
            <div class="metric-main">{avg_mrr:.4f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Average nDCG</div>
            <div class="metric-main">{avg_ndcg:.4f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Average Coverage</div>
            <div class="metric-main">{avg_coverage:.1f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Best MRR</div>
            <div class="metric-main">{best_mrr:.4f}</div>
        </div>
    </div>
    """


def build_metric_history(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["id", "metric", "value"])

    history = df.sort_values("id")
    rows = []
    for _, row in history.iterrows():
        rows.extend(
            [
                {"id": row["id"], "metric": "MRR", "value": row["mrr"]},
                {"id": row["id"], "metric": "nDCG", "value": row["ndcg"]},
                {"id": row["id"], "metric": "Coverage", "value": row["coverage"] / 100},
            ]
        )
    return pd.DataFrame(rows)


def get_experiment_ids(df: pd.DataFrame) -> list[int]:
    if df.empty:
        return []
    return df["id"].astype(int).tolist()


def load_selected_experiment(experiment_id: int | None) -> pd.DataFrame:
    if experiment_id is None or not table_exists():
        return pd.DataFrame(columns=["field", "value"])

    with get_sqldb_connection() as conn:
        cursor = conn.execute(
            f"SELECT * FROM {TABLE_NAME} WHERE id = ?",
            (experiment_id,),
        )
        row = cursor.fetchone()
        columns = [description[0] for description in cursor.description]

    if row is None:
        return pd.DataFrame(columns=["field", "value"])
    return pd.DataFrame({"field": columns, "value": list(row)})


def refresh_view(search: str, sort_label: str, sort_direction: str, limit: int):
    import gradio as gr

    df = load_experiments(search, sort_label, sort_direction, limit)
    ids = get_experiment_ids(df)
    selected_id = ids[0] if ids else None
    return (
        build_summary(df),
        df,
        build_metric_history(df),
        gr.update(choices=ids, value=selected_id),
        load_selected_experiment(selected_id),
    )


CSS = """
.gradio-container {
    max-width: 1280px !important;
}
.summary-grid {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 10px;
}
.metric-card {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 12px 14px;
    background: #ffffff;
}
.metric-label {
    color: #5b6472;
    font-size: 12px;
    margin-bottom: 5px;
}
.metric-main {
    color: #151922;
    font-size: 20px;
    font-weight: 700;
    overflow-wrap: anywhere;
}
@media (max-width: 900px) {
    .summary-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}
"""


def main():
    import gradio as gr

    with gr.Blocks(title="Retrieval Experiments") as app:
        gr.Markdown("# Retrieval Experiments")

        with gr.Row():
            search = gr.Textbox(
                label="Search",
                placeholder="embedding, retriever, reranker, prompt version",
                scale=3,
            )
            sort_label = gr.Dropdown(
                choices=list(SORT_COLUMNS.keys()),
                value="Newest",
                label="Sort",
                scale=1,
            )
            sort_direction = gr.Radio(
                choices=["Descending", "Ascending"],
                value="Descending",
                label="Order",
                scale=1,
            )
            limit = gr.Slider(
                minimum=10,
                maximum=1000,
                value=100,
                step=10,
                label="Rows",
                scale=1,
            )

        refresh = gr.Button("Refresh", variant="primary")
        summary = gr.HTML()

        with gr.Tab("Table"):
            experiments = gr.Dataframe(
                label="Experiments",
                interactive=False,
                wrap=True,
                max_height=420,
            )

        with gr.Tab("Trends"):
            metric_history = gr.LinePlot(
                x="id",
                y="value",
                color="metric",
                title="Retrieval Metrics by Experiment",
                y_lim=[0, 1],
                height=420,
            )

        with gr.Tab("Details"):
            experiment_id = gr.Dropdown(label="Experiment ID", choices=[], value=None)
            selected_experiment = gr.Dataframe(
                label="Selected Experiment",
                interactive=False,
                wrap=True,
                max_height=420,
            )

        inputs = [search, sort_label, sort_direction, limit]
        outputs = [summary, experiments, metric_history, experiment_id, selected_experiment]

        app.load(refresh_view, inputs=inputs, outputs=outputs)
        refresh.click(refresh_view, inputs=inputs, outputs=outputs)
        search.submit(refresh_view, inputs=inputs, outputs=outputs)
        experiment_id.change(
            load_selected_experiment,
            inputs=experiment_id,
            outputs=selected_experiment,
        )

    app.launch(
    inbrowser=True,
    css=CSS,
)


if __name__ == "__main__":
    main()
