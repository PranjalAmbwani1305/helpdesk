import os
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
        text += page.extract_text() + "\n"
    return text

# Function to store PDF content in Pinecone
def store_pdf_in_pinecone(pdf_name, pdf_text):
    vector = embedding_model.encode(pdf_text).tolist()
    index.upsert(vectors=[{"id": pdf_name, "values": vector, "metadata": {"filename": pdf_name}}])

# Function to fetch stored PDFs
def get_stored_pdfs():
    response = index.describe_index_stats()
    return response["total_vector_count"]

# Streamlit UI Layout
st.set_page_config(page_title="Legal HelpDesk", page_icon="⚖️", layout="wide")

# Header
st.markdown("<h1 style='text-align: center;'>⚖️ AI-Powered Legal HelpDesk for Saudi Arabia</h1>", unsafe_allow_html=True)
st.write("Helping you find legal information from Saudi Arabian laws quickly and accurately.")

# PDF Source Selection
st.subheader("📂 Select PDF Source")
pdf_source = st.radio("", ["Upload from PC", "Choose from the Document Storage"])

# File Upload Section
if pdf_source == "Upload from PC":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        pdf_text = extract_text_from_pdf(uploaded_file)
        store_pdf_in_pinecone(uploaded_file.name, pdf_text)
        st.success(f"✅ {uploaded_file.name} stored successfully!")

# Stored PDFs Section
elif pdf_source == "Choose from the Document Storage":
    st.subheader("📄 Stored Legal Documents")
    try:
        total_pdfs = get_stored_pdfs()
        st.write(f"📄 **Total PDFs Stored:** {total_pdfs}")
    except Exception as e:
        st.error(f"Error fetching stored PDFs: {e}")

# Language Selection
st.subheader("🌍 Choose Input Language")
input_language = st.radio("", ["English", "Arabic"], horizontal=True)

st.subheader("🌍 Choose Response Language")
response_language = st.radio("", ["English", "Arabic"], horizontal=True)

# Search Bar
st.markdown("## 🔍 Ask a question (in English or Arabic)")
query = st.text_input("Enter your legal question:")
if query:
    query_vector = embedding_model.encode(query).tolist()
    results = index.query(vector=query_vector, top_k=5, include_metadata=True)
    
    st.markdown("### 📜 Relevant Legal Documents:")
    for match in results["matches"]:
        st.write(f"📄 {match['metadata']['filename']} (Score: {match['score']:.2f})")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>Developed with ❤️ using Streamlit & Pinecone</p>", unsafe_allow_html=True)
