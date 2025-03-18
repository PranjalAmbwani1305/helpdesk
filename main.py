import os
import fitz  # PyMuPDF for PDF text extraction
import streamlit as st
import pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# Define index name
INDEX_NAME = "helpdesk"

# Check if index exists, else create it
if INDEX_NAME not in [index["name"] for index in pc.list_indexes()]:
    pc.create_index(INDEX_NAME, dimension=384, metric="cosine")

# Connect to Pinecone index
index = pc.Index(INDEX_NAME)

# Load sentence embedding model
hf_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Streamlit UI
st.title("📜 AI-Powered Legal HelpDesk for Saudi Arabia")
st.subheader("Upload PDFs and Ask Questions")

# File uploader (Multiple PDFs)
uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

def extract_text_from_pdf(uploaded_file):
    """Extracts text from an uploaded PDF file."""
    text_list = []
    pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")  # Open from bytes
    for page in pdf_document:
        text_list.append(page.get_text("text"))
    return "\n".join(text_list)

def store_text_in_pinecone(text, pdf_name):
    """Stores extracted text chunks in Pinecone"""
    sentences = text.split("\n")

    pinecone_data = []
    
    for i, sentence in enumerate(sentences):
        if sentence.strip():  # Skip empty lines
            vector = hf_model.encode(sentence).tolist()  # Convert to vector
            metadata = {
                "pdf_name": pdf_name,
                "text": sentence,
                "type": "article",
                "article_number": str(i+1)
            }
            pinecone_data.append({"id": f"{pdf_name}-article-{i}", "values": vector, "metadata": metadata})
    
    if pinecone_data:
        index.upsert(vectors=pinecone_data)
        st.write(f"✅ Stored {len(pinecone_data)} text chunks from {pdf_name} in Pinecone.")

def process_and_store_pdfs(uploaded_files):
    """Extracts text from PDFs and stores in Pinecone"""
    for uploaded_file in uploaded_files:
        text = extract_text_from_pdf(uploaded_file)
        store_text_in_pinecone(text, uploaded_file.name)
    st.success("📥 PDFs processed and stored in Pinecone successfully!")

if uploaded_files:
    process_and_store_pdfs(uploaded_files)

# Search query input
query = st.text_input("Ask a legal question (in English or Arabic):")

if query:
    query_vector = hf_model.encode(query).tolist()
    search_results = index.query(vector=query_vector, top_k=5, include_metadata=True)

    st.write(f"### 🔍 Search Results ({len(search_results['matches'])} hits)")
    
    for i, match in enumerate(search_results["matches"]):
        metadata = match["metadata"]
        st.write(f"#### {i+1}. {metadata['pdf_name']} - Article {metadata['article_number']}")
        st.write(f"📜 {metadata['text']}")
        st.write("---")
