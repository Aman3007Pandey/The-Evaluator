from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv
from src.embeddings import embeddings
from src.vector_store import vector_store
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, convert_to_messages
import gradio as gr
from langchain_core.documents import Document
from utility.rrf import reciprocal_rank_fusion_v2
from utility.cross_Encoder import cross_encoder_v1
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


load_dotenv(override=True)

MODEL_NAME = os.getenv("MODEL_NAME")
VECTOR_DB = os.getenv("VECTOR_DB")


RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", 5))  # Default to 5 if not set


# step 1 : Make vector store as retriever
retriever = vector_store.as_retriever()
# step 2: ChatOllama LLM
# llm=ChatOllama( model=MODEL_NAME,  
#                temperature=0)

# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     google_api_key=os.getenv("GOOGLE_API_KEY"),
#     temperature=0
# )

llm = ChatOpenAI(
    model="gpt-4.1-nano",
    temperature=0
)
# print(retriever.invoke("Who is Avery?"))
# print(llm.invoke("Who is Avery?"))

from src.vector_store import VECTOR_DB, vector_store,COLLECTION_NAME
import os

print("cwd:", os.getcwd())
print("VECTOR_DB:", VECTOR_DB)
print("abs path:", os.path.abspath(VECTOR_DB))
print("count:", vector_store._collection.count())
print(COLLECTION_NAME)

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
Your answer will be evaluated for accuracy, relevance and completeness, so make sure it only answers the question and fully answers it.
If you don't know the answer, say so.
For context, here are specific extracts from the Knowledge Base that might be directly relevant to the user's question:
{context}

With this context, please answer the user's question. Be accurate, relevant and complete.
"""

def fetch_context(question: str) -> list[Document]:
    """
    Retrieve relevant context documents for a question.
    """
    return retriever.invoke(question, k=RETRIEVAL_K)

def combined_question(question: str, history: list[dict] = []) -> str:
    """
    Combine all the user's messages into a single string.
    """
    prior = "\n".join(m["content"] for m in history if m["role"] == "user")
    return prior + "\n" + question


# def answerQuestion(question: str,history):
#     docs=fetch_context(question)
#     context = "\n\n".join(doc.page_content for doc in docs)
#     system_prompt=SYSTEM_PROMPT_TEMPLATE.format(context=context)
#     response = llm.invoke([SystemMessage(content=system_prompt),HumanMessage(content=question)])
#     return response.content



def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    """
    Answer the given question with RAG; return the answer and the context documents.
    """
    combined = combined_question(question, history)
    docs = fetch_context(combined)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = system_prompt.format(context=context)
    messages= [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    return response.content, docs

def answer_question_v2(question: str,bm25_index, history: list[dict] = []) -> tuple[str, list[Document]]:
    """
    Answer the given question with RAG; return the answer and the context documents.
    """
    combined = combined_question(question, history)
    retrieved_docs_old : list[Document]=reciprocal_rank_fusion_v2(combined,bm25_index)
    print(f"Retrieved {len(retrieved_docs_old)} documents for question: {question}")

    ## Pass them through ReRanker First
    docs = cross_encoder_v1(retrieved_docs_old,question)
    # docs = fetch_context(combined)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT.format(context=context)
    messages= [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    return response.content, docs



def answer_question_v3(question: str,bm25_index, history: list[dict] = []) -> tuple[str, list[Document]]:
    """
    Answer the given question with RAG; return the answer and the context documents.
    """
    chunks=reciprocal_rank_fusion_v2(question,bm25_index)
    new_chunks=cross_encoder_v1(chunks,question)
    context = "\n\n".join(
        f"Extract from {chunk.metadata['source']}:\n{chunk.page_content}"
        for chunk in new_chunks
    )

   
    system_prompt = SYSTEM_PROMPT.format(context=context)
    print(f"Length of context  is : {len(system_prompt)}")
    messages = [
        SystemMessage(content=system_prompt),
        *convert_to_messages(history),
        HumanMessage(content=question),
    ]

    response = llm.invoke(messages)

    return response.content, new_chunks

def generate_document_context(document: str, chunk: str) -> str:

    prompt = SYSTEM_PROMPT.format(
        document=document,
        chunk=chunk
    )

    response = llm.invoke(prompt)

    return response.content.strip()
# print(answerQuestion("Who is Avery?",[]) )

# gr.ChatInterface(answer_question).launch(inbrowser=True)