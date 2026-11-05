# from langchain_chroma import Chroma
# from src.embeddings import embeddings
# from dotenv import load_dotenv
# import os

# load_dotenv(override=True)
# VECTOR_DB=os.getenv("VECTOR_DB", "chroma")
# COLLECTION_NAME=os.getenv("COLLECTION_NAME", "knowledge_base")

# print(f"Using vector database directory: {VECTOR_DB}")
# print(f"Using collection name: {COLLECTION_NAME}")

# vector_store = Chroma(persist_directory=VECTOR_DB, embedding_function=embeddings,
#                       collection_name=COLLECTION_NAME)


from pathlib import Path
import os

from dotenv import load_dotenv
from langchain_chroma import Chroma

from src.embeddings import embeddings

# Load environment variables
load_dotenv(override=True)

# Project root (assuming this file is in src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Read values from .env
VECTOR_DB = os.getenv("VECTOR_DB", "chroma")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "knowledge_base")

# Build an absolute path to the vector database
VECTOR_DB_PATH = PROJECT_ROOT / VECTOR_DB

print(f"Using vector database directory: {VECTOR_DB_PATH}")
print(f"Using collection name: {COLLECTION_NAME}")




# Create/load the Chroma vector store
vector_store = Chroma(
    persist_directory=str(VECTOR_DB_PATH),
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME,
)
# vector_store.reset_collection()
# print("vector store cleared")

# Optional debug information
try:
    print(f"Number of documents in collection: {vector_store._collection.count()}")
except Exception as e:
    print(f"Could not determine collection count: {e}")