from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# print(os.getenv("OPENAI_API_KEY"))

# llm = ChatOpenAI(
#     model="gpt-4.1-nano",
#     temperature=0
# )

# response = llm.invoke("Who is Avery?")
# print(response.content)


from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

for model in client.models.list().data:
    print(model.id)