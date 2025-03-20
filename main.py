import os
import re
import pinecone
import streamlit as st
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

# Load environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "helpdesk"

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Load SentenceTransformer model from Hugging Face
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text + "\n"
    return text

# Function to store PDF content in Pinecone
def store_pdf_in_pinecone(pdf_name, pdf_text):
    try:
        vector = embedding_model.encode(pdf_text).tolist()
        
        # Store with metadata (Ensure filename is stored)
        index.upsert(vectors=[
            {
                "id": pdf_name,
                "values": vector,
                "metadata": {"filename": pdf_name}  # Ensure filename is stored
            }
        ])
        return True
    except Exception as e:
        st.error(f"Error storing PDF in Pinecone: {e}")
        return False

# Function to fetch and clean stored PDF names
def get_stored_pdf_names():
    try:
        response = index.describe_index_stats()
        total_pdfs = response.get("total_vector_count", 0)
        
        # Fetch all stored items (use correct metadata keys)
        query_results = index.query(vector=[0] * 384, top_k=total_pdfs, include_metadata=True)
        
        pdf_names = []
        for match in query_results["matches"]:
            metadata = match.get("metadata", {})  # Ensure metadata exists
            
            # Fetch filename safely
            filename = metadata.get("filename", "Unknown PDF")  
            
            # Remove file extension (.pdf) and clean URL-like names
            clean_name = re.sub(r'^www\.', '', filename)  # Remove 'www.'
            clean_name = clean_name.replace(".pdf", "")   # Remove '.pdf'
            
            pdf_names.append(clean_name)

        return pdf_names
    except Exception as e:
        st.error(f"Error fetching stored PDFs: {e}")
        return []

# Streamlit UI Layout
st.set_page_config(page_title="Legal HelpDesk", page_icon="⚖️", layout="wide")

# Header
st.title("AI-Powered Legal HelpDesk for Saudi Arabia")
st.write("Helping you find legal information from Saudi Arabian laws quickly and accurately.")

# PDF Source Selection
st.subheader("Select PDF Source")
pdf_source = st.radio("Choose PDF Source:", ["Upload from PC", "Choose from the Document Storage"])

# File Upload Section
if pdf_source == "Upload from PC":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        pdf_text = extract_text_from_pdf(uploaded_file)
        success = store_pdf_in_pinecone(uploaded_file.name, pdf_text)
        if success:
            st.success(f"{uploaded_file.name} stored successfully!")

# Stored PDFs Section
elif pdf_source == "Choose from the Document Storage":
    st.subheader("📁 Stored Legal Documents")
    pdf_names = get_stored_pdf_names()
    
    if pdf_names:
        for pdf_name in pdf_names:
            st.write(f"📑 {pdf_name}")  # Display each cleaned PDF name
    else:
        st.write("No PDFs stored yet.")
