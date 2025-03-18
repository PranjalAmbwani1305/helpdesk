import os
import fitz  # PyMuPDF
import streamlit as st
import pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "helpdesk"
index = pc.Index(INDEX_NAME)

# Load embedding model
hf_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Streamlit UI
st.title("📜 AI-Powered Legal HelpDesk for Saudi Arabia")
st.subheader("Upload PDFs and Ask Legal Questions")

# File uploader (Multiple PDFs)
uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

def extract_text_from_pdf(uploaded_file):
    """Extract text from an uploaded PDF file."""
    text_list = []
    pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")  # Open from bytes
    for page in pdf_document:
        text_list.append(page.get_text("text"))
    return "\n".join(text_list)

def store_text_in_pinecone(text, pdf_name):
    """Stores extracted text chunks in Pinecone"""
    sentences = text.split("\n")
    
    for i, sentence in enumerate(sentences):
        if sentence.strip():  # Skip empty lines
            vector = hf_model.encode(sentence).tolist()  # Convert to vector
            metadata = {
                "pdf_name": pdf_name,
                "text": sentence,
                "type": "article",
                "article_number": str(i+1)
            }
            index.upsert([(f"{pdf_name}-article-{i}", vector, metadata)])

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
    search_results = index.query(query_vector, top_k=5, include_metadata=True)

    st.write(f"### 🔍 Search Results ({len(search_results['matches'])} hits)")
    
    for i, match in enumerate(search_results["matches"]):
        metadata = match["metadata"]
        st.write(f"#### {i+1}. {metadata['pdf_name']} - Article {metadata['article_number']}")
        st.write(f"**Text:** {metadata['text']}")
        st.write(f"**Score:** {match['score']:.4f}")
        st.write("---")
