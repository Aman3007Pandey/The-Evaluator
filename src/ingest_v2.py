from glob import glob
import os
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.vector_store import vector_store
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from src.embeddings import embeddings
from src.llm import generate_document_context
load_dotenv(override=True)

chunk_size = int(os.getenv("CHUNK_SIZE",500))
chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 200))
chunking_strat=os.getenv("CHUNKING_STRAT")

# print(f"Chunk size: {chunk_size}")
# print(f"Chunk overlap: {chunk_overlap}")

prefix_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(prefix_path)
knowledge_base_path = os.path.join(prefix_path, "knowledge-base")
print(knowledge_base_path)


# find all markdown files in the knowledge base recursively

markdown_files = glob(knowledge_base_path + "/**/*.md", recursive=True)
#print(f"Knowledge base path: {knowledge_base_path}")
print(f"Found {len(markdown_files)} markdown files.")


documents = []
document_lookup = {} # So that current chunk and get full context of document 
# key : file_path = value : file content


for idx, file_path in enumerate(markdown_files):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        # create a document with the content and the file path as metadata
        doc = Document(page_content=content, metadata={"source": file_path})
        # print(f"Created document for {file_path} with content length {len(content)}.")
        documents.append(doc)
        document_lookup[file_path]=content # Building doc lookup

print(f"Successfully created LangChain documents for all markdown files. Total documents: {len(documents)}.")        

print(f"Chunking Statrgy Used : {chunking_strat} ", end="\n")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
chunks=text_splitter.split_documents(documents)


for idx, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"] = idx

## Calling LLM for every chunk process will be slow 
for chunk in chunks:
    full_document = document_lookup[chunk.metadata["source"]]
    print("original")
    print("-----------------------------------------------------------")
    print(chunk.page_content)  
    final_context = generate_document_context(full_document,chunk.page_content)
    chunk.page_content = f"""
        Context:
        {final_context}

        Content:
        {chunk.page_content}
        """   
    print("-----------------------------------------------------------")
    print("After LLM transformation")
    print(chunk.page_content)    

print(f"Created {len(chunks)} chunks.")

vector_store.reset_collection()
print("vector store cleared")
vector_store.add_documents(chunks)
print(f"Successfully added {len(chunks)} chunks to the vector store.")