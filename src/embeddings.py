from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv(override=True)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

# embeddings=OllamaEmbeddings(model=EMBEDDING_MODEL)



embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)