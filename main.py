import os
import pinecone
import streamlit as st
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

# Load Pinecone API Key and Index Name
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "helpdesk"

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Load SentenceTransformer model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Function to get stored PDF names from Pinecone
def get_stored_pdfs():
    try:
        results = index.describe_index_stats()
        if "namespaces" in results:
            return list(results["namespaces"].keys())
        return []
    except Exception as e:
        st.error(f"Error fetching PDFs from Pinecone: {e}")
        return []

# Streamlit UI
st.title("📖 Legal HelpDesk for Saudi Arabia")

# Section: Select PDF Source
st.header("📑 Select PDF Source")

pdf_source = st.radio("Choose a source:", ["Upload from PC", "Choose from the Document Storage"])

if pdf_source == "Upload from PC":
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf", help="Limit: 200MB per file")
    
    if uploaded_file:
        st.success(f"Uploaded: {uploaded_file.name}")
        # Process the uploaded file (Extract and store in Pinecone)

elif pdf_source == "Choose from the Document Storage":
    # Fetch stored PDFs from Pinecone
    stored_pdfs = get_stored_pdfs()
    
    if stored_pdfs:
        selected_pdf = st.selectbox("Select a PDF", stored_pdfs)
        st.success(f"Selected: {selected_pdf}")
    else:
        st.warning("No PDFs found in storage.")

# Choose Input & Response Language
st.header("🌐 Language Settings")
input_language = st.radio("Choose Input Language:", ["English", "Arabic"])
response_language = st.radio("Choose Response Language:", ["English", "Arabic"])

# Ask Question
st.header("💬 Ask a Question")
question = st.text_input("Enter your question (in English or Arabic):")

if st.button("Submit"):
    if question:
        st.write(f"Searching for answers related to: **{question}**")
    else:
        st.warning("Please enter a question before submitting.")
