import streamlit as st
import openai
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
HF_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))

os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "My Groq Q&A chatbot"# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful friend. Please respond to the user."),
    ("user", "Question: {question}")
])
def gen_response(question, model_name, temperature, max_tokens):
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question': question})
    return answer

st.title("Basic Q&A Chatbot with Groq & HF - Rohan")

st.sidebar.title("Settings")
model_name = st.sidebar.selectbox("Choose a Groq Model", [
    "Llama3-8b-8192",
    "Mixtral-8x7b-32768",
    "Gemma-7b-it"
])
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5)
max_token = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=100)

st.write("Let's go, ask me anything!")
query = st.text_input("You:")

if query :
    res = gen_response(query, model_name, temperature, max_token)
    st.write(res)
else:
    st.write("Looks like you haven't asked anything yet!")
