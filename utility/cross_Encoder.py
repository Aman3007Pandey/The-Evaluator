from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

k=int(os.getenv("RERANKER_RETRIEVAL_K"))
crossEncoderName=os.getenv("RERANKER_METHOD")
# model = CrossEncoder( "cross-encoder/ms-marco-MiniLM-L-6-v2")
model = CrossEncoder(crossEncoderName)

def cross_encoder_v1 (retrieved_docs:list[Document],query:str):
    # Load model once per startup or else system is slow 
    # model = CrossEncoder( "cross-encoder/ms-marco-MiniLM-L-6-v2")


    pairs = [
    [query, doc.page_content]
    for doc in retrieved_docs
]

    scores = model.predict(pairs)
    
    scored_docs=list(zip(retrieved_docs,scores))
    ranked_docs=sorted(
        scored_docs,
        key= lambda x :x[1],
        reverse=True
    )
    ranked_docs=ranked_docs[:k]
    return [doc for doc,score in ranked_docs]