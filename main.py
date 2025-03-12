import streamlit as st
import pinecone
import os
import tempfile
import fitz  # PyMuPDF for extracting text from PDFs

from pinecone import Pinecone
# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "helpdesk"

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text("text") + "\n"
    except Exception as e:
        st.error(f"Error extracting text: {e}")
    return text

# Store PDF metadata in Pinecone
def store_pdf_in_pinecone(pdf_name, text_content):
    vector = [0] * 512  # Placeholder vector, replace with real embeddings
    index.upsert([(pdf_name, vector, {"filename": pdf_name, "content": text_content})])

# Retrieve stored PDFs from Pinecone
def get_stored_pdfs():
    stored_pdfs = []
    try:
        query_result = index.query(vector=[0] * 512, top_k=10, include_metadata=True)  
        for match in query_result.get('matches', []):
            if 'filename' in match['metadata']:
                stored_pdfs.append(match['metadata']['filename'])
    except Exception as e:
        st.error(f"🔴 Error retrieving PDFs: {e}")
    return stored_pdfs

# Streamlit UI Setup
st.set_page_config(page_title="AI-Powered Legal HelpDesk", layout="wide")
st.title("🛡️ AI-Powered Legal HelpDesk for Saudi Arabia")

# Sidebar - Stored PDFs
st.sidebar.title("📂 Stored PDFs")
stored_pdfs = get_stored_pdfs()
if stored_pdfs:
    for pdf in stored_pdfs:
        st.sidebar.write(f"📄 {pdf}")
 else:
     st.sidebar.write("No PDFs stored yet.")

# PDF Upload Section
st.subheader("📄 Upload or Select a PDF")
pdf_source = st.radio("Select PDF Source", ["Upload from PC", "Choose from the Document Storage"])

if pdf_source == "Upload from PC":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"✅ {uploaded_file.name} uploaded successfully!")

        # Extract text & store in Pinecone
        pdf_text = extract_text_from_pdf(file_path)
        store_pdf_in_pinecone(uploaded_file.name, pdf_text)

# Language Selection
st.subheader("🌍 Choose Input & Response Language")
input_language = st.radio("Choose Input Language", ["English", "Arabic"])
response_language = st.radio("Choose Response Language", ["English", "Arabic"])

# Question Input
st.subheader("❓ Ask a legal question:")
user_query = st.text_input("Enter your legal query...")

# Query Processing (Placeholder for AI model)
if st.button("🔍 Submit"):
    if user_query:
        st.success("🧐 Searching for relevant legal information...")
        # Placeholder AI response
        response = "🔹 AI-generated response based on legal documents."
        st.write(response)
    else:
        st.warning("⚠️ Please enter a question.")
