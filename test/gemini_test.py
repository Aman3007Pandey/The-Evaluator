from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv(override=True)
print(os.getenv("GEMINI_API_KEY"))

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    GEMINI_API_KEY=os.getenv("GEMINI_API_KEY"),
    temperature=0
)

print(llm.invoke("Who is Avery?"))