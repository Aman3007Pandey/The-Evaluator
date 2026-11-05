from src.vector_store import vector_store
from utility.bm25_build import build_BM25_index
from utility.bm25_search import bm25_search
from dotenv import load_dotenv
import os 

load_dotenv(override=True)

retrieval_k= int(os.getenv("RETRIEVAL_K"))

bm25_index=build_BM25_index(vector_store)

bm25_search("Who Won the IIOTY Award ?" , bm25_index,vector_store, retrieval_k)
