from dotenv import load_dotenv
import re
import os 
load_dotenv(override=True)

retrieval_k= os.getenv("RETRIEVAL_K")

def bm25_search(query, bm25_index, vector_store, retrieval_k=5):

    data = vector_store.get()
    documents = data["documents"]
    metadatas = data["metadatas"]

    query_tokens = re.findall(r"\w+", query.lower())

    scores = bm25_index.get_scores(query_tokens)
    indexed_scores = list(enumerate(scores))

    ranked = sorted(
    indexed_scores,
    key=lambda x: x[1],
    reverse=True
    )

    top_results = ranked[:retrieval_k]
    results=[]

    for idx, score in top_results:

        results.append({
            "chunk_id": metadatas[idx]["chunk_id"],
            "document": documents[idx],
            "score": score
        })

    return results


