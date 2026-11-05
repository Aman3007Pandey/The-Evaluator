import os
from dotenv import load_dotenv
from evaluations.eval import evaluate_all_retrieval,evaluate_all_retrieval_v2
from src.database import  get_sqldb_connection
from evaluations.eval import TestQuestion, RetrievalEval
from utility.bm25_build import build_BM25_index
load_dotenv(override=True)
from src.vector_store import vector_store
import time


def _run_retrieval_evaluation() -> dict[str, float | int]:
    total_mrr = 0.0
    total_ndcg = 0.0
    total_coverage = 0.0
    count = 0

    ## Call build BM25_index only once
    bm25_index=build_BM25_index(vector_store)

    startTime=time.time()
    ## Invoking multiprocessing one
    for _test, result, progress in evaluate_all_retrieval_v2(bm25_index):
        count += 1
        total_mrr += result.mrr
        total_ndcg += result.ncdg
        total_coverage += result.keyword_coverage
        print(f"Evaluated retrieval test {count} ({progress:.0%})")

    totalTimeTaken=time.time()-startTime
    if count == 0:
        raise ValueError("No retrieval tests found to evaluate.")

    print(total_mrr, total_ndcg, total_coverage, count)
    return {
        "test_count": count,
        "mrr": total_mrr / count,
        "ndcg": total_ndcg / count,
        "coverage": total_coverage / count,
        "time_taken":totalTimeTaken
    }


def log_retrieval_experiment() -> int:
    """Run retrieval evaluation and log the aggregate result to SQLite."""
    metrics = _run_retrieval_evaluation()

    conn = get_sqldb_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO retrieval_experiments (
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
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?,?,?)
            """,
            (
                os.getenv("EMBEDDING_MODEL"),
                os.getenv("RETRIEVER_METHOD"),
                os.getenv("RERANKER_METHOD"),
                os.getenv("PROMPT_VERSION"),
                int(os.getenv("CHUNK_SIZE")),
                int(os.getenv("CHUNK_OVERLAP")),
                int(os.getenv("RETRIEVAL_K")),
                metrics["mrr"],
                metrics["ndcg"],
                metrics["coverage"],
                metrics["time_taken"],
                os.getenv("CHUNKING_STRAT")

            ),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


if __name__ == "__main__":
    experiment_id = log_retrieval_experiment()
    print(f"Logged retrieval experiment id: {experiment_id}")
