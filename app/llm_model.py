from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage

os.environ["GOOGLE_API_KEY"] = "AIzaSyApkATVF4lSyCcom-aFzmxiuCzgcFtm6bQ"

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  
    temperature=0.7
)
