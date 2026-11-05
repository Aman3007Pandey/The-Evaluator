from src.vector_store import vector_store
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

load_dotenv(override=True)
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", 5))  # Default to 5 if not set

retriever = vector_store.as_retriever()

def fetch_context(question: str) -> list[Document]:
    """
    Retrieve relevant context documents for a question.
    """
    return retriever.invoke(question, k=RETRIEVAL_K)

