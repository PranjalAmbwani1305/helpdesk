import os
import streamlit as st
import pinecone
import fitz  # PyMuPDF for PDF processing
import numpy as np
from sentence_transformers import SentenceTransformer

# --- Initialize Pinecone ---
pinecone.init(api_key="your-pinecone-api-key", environment="your-environment")
index = pinecone.Index("your-index-name")

# --- Load Embedding Model ---
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Function to Extract Text from PDF ---
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text.strip()

# --- Function for Text Chunking ---
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks

# --- Function to Store Vectors in Pinecone ---
def store_vectors(chunks, pdf_name):
    for i, chunk in enumerate(chunks):
        vector = embedding_model.encode(chunk)  # Get embeddings
        vector = np.array(vector).tolist()  # Ensure 1D list of floats
        
        index.upsert([
            (f"{pdf_name}-doc-{i}", vector, {"pdf_name": pdf_name, "text": chunk})
        ])

# --- Function to Retrieve Relevant Chunks ---
def search_similar_chunks(query):
    query_vector = embedding_model.encode(query).tolist()
    results = index.query(query_vector, top_k=5, include_metadata=True)

    return [match["metadata"]["text"] for match in results["matches"]]

# --- Streamlit UI ---
st.title("AI-Powered Legal HelpDesk 📜🤖")

uploaded_file = st.file_uploader("Upload a Legal Document (PDF)", type=["pdf"])

if uploaded_file:
    st.success("Processing PDF...")
    
    temp_pdf_path = f"temp_{uploaded_file.name}"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    text = extract_text_from_pdf(temp_pdf_path)
    chunks = chunk_text(text)
    store_vectors(chunks, uploaded_file.name)

    st.success("PDF successfully processed and stored in Pinecone!")

query = st.text_input("Ask a Legal Question:")
if query:
    results = search_similar_chunks(query)
    st.subheader("Relevant Information:")
    for res in results:
        st.write("- " + res)
