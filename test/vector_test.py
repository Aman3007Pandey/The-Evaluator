from langchain_chroma import Chroma
from src.embeddings import embeddings
from dotenv import load_dotenv
import os
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go


## How to Run : python -m test.vector_test
load_dotenv(override=True)

VECTOR_DB = os.getenv("VECTOR_DB")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=VECTOR_DB,
)

# Get underlying Chroma collection
collection = vector_store._collection

# Fetch all embeddings, documents and metadata
result = collection.get(
    include=["embeddings", "documents", "metadatas"]
)

vectors = np.array(result["embeddings"])
documents = result["documents"]
metadatas = result["metadatas"]

# Infer document type from source path
doc_types = []
for metadata in metadatas:
    source = metadata.get("source", "")

    if "products" in source:
        doc_types.append("products")
    elif "employees" in source:
        doc_types.append("employees")
    elif "contracts" in source:
        doc_types.append("contracts")
    elif "company" in source:
        doc_types.append("company")
    else:
        doc_types.append("unknown")

color_map = {
    "products": "blue",
    "employees": "green",
    "contracts": "red",
    "company": "orange",
    "unknown": "gray",
}

colors = [color_map[t] for t in doc_types]

# Reduce dimensions
tsne = TSNE(n_components=3, random_state=42)
reduced_vectors = tsne.fit_transform(vectors)

# Plot
fig = go.Figure(
    data=[
        go.Scatter3d(
            x=reduced_vectors[:, 0],
            y=reduced_vectors[:, 1],
            z=reduced_vectors[:, 2],
            mode="markers",
            marker=dict(size=5, color=colors, opacity=0.8),
            text=[
                f"Type: {t}<br>Source: {m.get('source')}<br>{d[:100]}..."
                for t, m, d in zip(doc_types, metadatas, documents)
            ],
            hoverinfo="text",
        )
    ]
)

fig.update_layout(
    title="Chroma Vector Store Visualization",
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z",
    ),
    width=900,
    height=700,
)

fig.show()