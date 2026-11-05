from collections import defaultdict
from utility.bm25_search import bm25_search
from dotenv import load_dotenv
import re
import os 
from utility.bm25_build import build_BM25_index
from src.vector_store import vector_store
from langchain_core.documents import Document

load_dotenv(override=True)

retrieval_k= int(os.getenv("RETRIEVAL_K"))

#  Call this function build_BM25_index only once in PROD
def reciprocal_rank_fusion (query) -> list[Document]:
    
    from src.llm import fetch_context
    k=60 # Smoothing Constant
    bm25_index=build_BM25_index(vector_store)
    bm25_results=bm25_search(query , bm25_index,vector_store, retrieval_k)
    vector_results = fetch_context(query)

    # print(type(bm25_results))
    # print(bm25_results)
    # print(vector_results[0],end="\n")
    # print("---------------------------------------------------------------------------------")
    # print(vector_results[0].metadata)
    # print(vector_store.get()["metadatas"][0])
    # print(vector_store.get()["metadatas"][1])
    # print(vector_store.get()["metadatas"][2])

    # Build lookup table
    data = vector_store.get()

    # print("---------------------------------------------------------------------------------")
    # print(data["ids"][:10])
    # print("---------------------------------------------------------------------------------")
    # print(data["metadatas"][:3])
    # print("---------------------------------------------------------------------------------")
    # print(vector_results[0])
    # print("---------------------------------------------------------------------------------")
    # print(vector_results[0].metadata)
    # print("---------------------------------------------------------------------------------")

    rrf_scores = defaultdict(float)
    for rank, result in enumerate(bm25_results, start=1):

        doc_id = result["chunk_id"]
        rrf_scores[doc_id] += 1 / (k + rank)

    for rank, doc in enumerate(vector_results, start=1):

        doc_id = doc.metadata["chunk_id"]
        rrf_scores[doc_id] += 1 / (k + rank)  

    ranked_results = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    ) 

     

    chunk_lookup = {
    metadata["chunk_id"]: Document(
        page_content=document,
        metadata=metadata
    )
    for metadata, document in zip(
        data["metadatas"],
        data["documents"]
    )
    }

    # print(chunk_lookup[319])

    return [
    chunk_lookup[chunk_id]
    for chunk_id, score in ranked_results[:retrieval_k]
    ]       

# print(reciprocal_rank_fusion("Who Won the IIOTY Award ?")[0])
# # print("----------------------------------------------------")
# # print(reciprocal_rank_fusion("Who Won the IIOTY Award ?")[1])


def reciprocal_rank_fusion_v2 (query,bm25_index) -> list[Document]:
    
    from src.llm import fetch_context
    k=60 # Smoothing Constant
    bm25_results=bm25_search(query , bm25_index,vector_store, retrieval_k)
    vector_results = fetch_context(query)

    # print(type(bm25_results))
    # print(bm25_results)
    # print(vector_results[0],end="\n")
    # print("---------------------------------------------------------------------------------")
    # print(vector_results[0].metadata)
    # print(vector_store.get()["metadatas"][0])
    # print(vector_store.get()["metadatas"][1])
    # print(vector_store.get()["metadatas"][2])

    # Build lookup table
    data = vector_store.get()

    # print("---------------------------------------------------------------------------------")
    # print(data["ids"][:10])
    # print("---------------------------------------------------------------------------------")
    # print(data["metadatas"][:3])
    # print("---------------------------------------------------------------------------------")
    # print(vector_results[0])
    # print("---------------------------------------------------------------------------------")
    # print(vector_results[0].metadata)
    # print("---------------------------------------------------------------------------------")

    rrf_scores = defaultdict(float)
    for rank, result in enumerate(bm25_results, start=1):

        doc_id = result["chunk_id"]
        rrf_scores[doc_id] += 1 / (k + rank)

    for rank, doc in enumerate(vector_results, start=1):

        doc_id = doc.metadata["chunk_id"]
        rrf_scores[doc_id] += 1 / (k + rank)  

    ranked_results = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    ) 

     

    chunk_lookup = {
    metadata["chunk_id"]: Document(
        page_content=document,
        metadata=metadata
    )
    for metadata, document in zip(
        data["metadatas"],
        data["documents"]
    )
    }

    # print(chunk_lookup[319])

    return [
    chunk_lookup[chunk_id]
    for chunk_id, score in ranked_results[:retrieval_k]
    ]