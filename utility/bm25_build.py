from dotenv import load_dotenv
import re
from rank_bm25 import BM25Okapi
load_dotenv(override=True)




def build_BM25_index(vector_store):
    # Placeholder for building BM25 index logic
    print("Building BM25 index...")

    data = vector_store.get()
    # print(data.keys())
    documents = data['documents']
    tokenized_documents = []
    for doc in documents:
        tokenized_documents.append(re.findall(r'\w+', doc.lower()))

    bm25_index = BM25Okapi(tokenized_documents)

    # Implement the actual BM25 indexing logic here
    print("BM25 index built successfully.")
    return bm25_index

