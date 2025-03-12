import os
import streamlit as st
import fitz  # PyMuPDF
import pinecone
import hashlib
import json
import asyncio
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch

# Set up Pinecone API
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
index_name = "helpdesk"

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(index_name)

# Ensure Async Event Loop Setup
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Load Hugging Face Model for Embeddings
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


# Function to get text embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()


# Function to process PDF and extract text
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text


# Function to upload and store multiple PDFs in Pinecone
def process_and_store_pdf(uploaded_file):
    if uploaded_file is not None:
        pdf_name = uploaded_file.name
        file_path = os.path.join("/tmp", pdf_name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        pdf_text = extract_text_from_pdf(file_path)

        # Generate unique ID using hash
        pdf_id = hashlib.md5(pdf_name.encode()).hexdigest()
        vector = get_embedding(pdf_text)

        # Store in Pinecone
        index.upsert(vectors=[(pdf_id, vector, {"pdf_name": pdf_name, "content": pdf_text})])

        st.success(f"PDF '{pdf_name}' uploaded and processed successfully!")


# Function to retrieve stored PDFs
def get_stored_pdfs():
    query_results = index.describe_index_stats()
    vector_count = query_results["total_vector_count"]

    stored_pdfs = []
    if vector_count > 0:
        for namespace, details in query_results["namespaces"].items():
            if "vector_count" in details and details["vector_count"] > 0:
                stored_pdfs.append(namespace)

    return stored_pdfs


# UI: Sidebar for Uploaded PDFs
st.sidebar.title("📂 Stored PDFs")
stored_pdfs = get_stored_pdfs()
selected_pdf = st.sidebar.selectbox("Select a PDF", stored_pdfs if stored_pdfs else ["No PDFs Found"], key="pdf_dropdown_unique")

# UI: Main Page
st.title("📜 AI-Powered Legal HelpDesk")

st.subheader("Select PDF Source")
upload_option = st.radio("Choose an option:", ["Upload from PC", "Choose from Document Storage"])

if upload_option == "Upload from PC":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], key="pdf_uploader")
    if uploaded_file:
        process_and_store_pdf(uploaded_file)

st.subheader("Ask a legal question:")
query = st.text_area("Type your question here:")

if st.button("Get Answer"):
    if selected_pdf and selected_pdf != "No PDFs Found":
        query_vector = get_embedding(query)

        # Query Pinecone
        results = index.query(queries=[query_vector], top_k=5, include_metadata=True)
        answer = results["matches"][0]["metadata"]["content"] if results["matches"] else "No relevant information found."

        st.write("### AI Answer:")
        st.write(answer)
    else:
        st.error("Please select a PDF before asking a question.")
