import streamlit as st
import pinecone
import pdfplumber
import os
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator

# Load API Keys from Streamlit Secrets
HUGGINGFACE_API_KEY = st.secrets["HUGGINGFACE_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENV"]
PINECONE_INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]


if PINECONE_INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(PINECONE_INDEX_NAME, dimension=1536, metric="cosine")
index = pinecone.Index(PINECONE_INDEX_NAME)

# Load Hugging Face Model for Embeddings
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Streamlit App UI
st.set_page_config(page_title="AI-Powered Legal HelpDesk", layout="wide")
st.markdown("<h1 style='text-align: center;'>📜 AI-Powered Legal HelpDesk</h1>", unsafe_allow_html=True)

# Sidebar for PDF Upload
st.sidebar.header("📂 Upload & Stored PDFs")
uploaded_file = st.sidebar.file_uploader("Upload a legal document (PDF)", type=["pdf"])

# Function to Extract Text from PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Function to Check if PDF Exists in Pinecone
def pdf_exists(pdf_name):
    query_results = index.query(vector=[0]*1536, top_k=1, include_metadata=True, filter={"pdf_name": {"$eq": pdf_name}})
    return len(query_results["matches"]) > 0

# Function to Store PDF in Pinecone
def store_pdf_in_pinecone(pdf_name, text):
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    for i, chunk in enumerate(chunks):
        vector = model.encode(chunk).tolist()
        index.upsert([(f"{pdf_name}-chunk-{i}", vector, {"pdf_name": pdf_name, "text": chunk})])

# Process PDF if Uploaded
if uploaded_file:
    pdf_text = extract_text_from_pdf(uploaded_file)
    pdf_name = uploaded_file.name
    
    if not pdf_exists(pdf_name):
        store_pdf_in_pinecone(pdf_name, pdf_text)
        st.sidebar.success("✅ PDF uploaded and stored successfully!")
    else:
        st.sidebar.info("📌 PDF already exists in the database.")

# Input Query
query = st.text_input("🔍 Ask a legal question:")

if st.button("Get Answer"):
    if query and uploaded_file:
        query_embedding = model.encode(query).tolist()
        results = index.query(query_embedding, top_k=5, include_metadata=True, filter={"pdf_name": {"$eq": uploaded_file.name}})
        
        if results["matches"]:
            st.write("### 📝 Relevant Legal Information:")
            for match in results["matches"]:
                st.write(f"- {match['metadata']['text']}")
        else:
            st.warning("⚠️ No relevant information found in the selected document.")
    else:
        st.warning("📌 Please upload a PDF and enter a question.")
