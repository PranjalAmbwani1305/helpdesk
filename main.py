import os
import pinecone
from pinecone import Pinecone
import streamlit as st
import PyPDF2
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import login

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
index_name = "helpdesk"

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(index_name)


embedder = HuggingFaceEmbeddings(model_name="mistralai/Mistral-7B-Instruct-v0.1")

def process_pdf(pdf_path, chunk_size=500):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def store_vectors(chunks, pdf_name):
    for i, chunk in enumerate(chunks):
        vector = embedder.embed([chunk])[0]
        index.upsert([(f"{pdf_name}-doc-{i}", vector, {"pdf_name": pdf_name, "text": chunk})])

def query_vectors(query):
    vector = embedder.embed([query])[0]
    results = index.query(vector=vector, top_k=5, include_metadata=True)
    return results

st.markdown("# AI-Powered Legal HelpDesk")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    temp_pdf_path = f"temp_{uploaded_file.name}"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.read())
    chunks = process_pdf(temp_pdf_path)
    store_vectors(chunks, uploaded_file.name)
    st.success("PDF processed and indexed!")

query = st.text_input("Ask a question:")

if st.button("Get Answer") and query:
    results = query_vectors(query)
    if results["matches"]:
        st.write("**Answer:**", results["matches"][0]["metadata"]["text"])
    else:
        st.warning("No relevant information found.")
