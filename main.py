import streamlit as st
import pinecone
import pdfplumber
from sentence_transformers import SentenceTransformer
import os
import json

# Load API keys from secrets
HUGGINGFACE_API_KEY = st.secrets["HUGGINGFACE_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENV"]
PINECONE_INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index(PINECONE_INDEX_NAME)

# Load Sentence Transformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

st.set_page_config(page_title="AI-Powered Legal Helpdesk", layout="wide")
st.markdown("<h1 style='text-align: center;'>AI-Powered Legal Helpdesk</h1>", unsafe_allow_html=True)

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    return text

# Function to split text into chunks
def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Sidebar - Upload PDF
pdf_file = st.sidebar.file_uploader("Upload a legal document (PDF)", type=["pdf"])

if pdf_file:
    doc_text = extract_text_from_pdf(pdf_file)
    chunks = chunk_text(doc_text)
    
    st.sidebar.success("PDF uploaded and processed successfully!")
    
    # Store embeddings in Pinecone
    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk).tolist()
        index.upsert([(f"{pdf_file.name}-chunk-{i}", embedding, {"text": chunk})])
        
    st.sidebar.info("Document indexed successfully!")
else:
    chunks = []

# Query input
query = st.text_input("Ask your legal question:")

if st.button("Get Answer") and query:
    query_embedding = model.encode(query).tolist()
    result = index.query(query_embedding, top_k=5, include_metadata=True)
    
    if "matches" in result and result["matches"]:
        st.write("### Relevant Legal Information:")
        for match in result["matches"]:
            st.write(f"- {match['metadata']['text']}")
    else:
        st.warning("No relevant information found.")
