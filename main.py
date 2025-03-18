import os
import pinecone
import streamlit as st
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

# Load environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "helpdesk"

# Initialize Pinecone (NEW FIX)
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Load SentenceTransformer model from Hugging Face
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PdfReader(pdf_file)
        text = "\n".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
        return text if text else "No text found in this PDF."
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

# Function to store PDF content in Pinecone
def store_pdf_in_pinecone(pdf_name, pdf_text):
    try:
        vector = embedding_model.encode(pdf_text).tolist()
        index.upsert([(pdf_name, vector, {"filename": pdf_name})])
        return True
    except Exception as e:
        st.error(f"Error storing PDF in Pinecone: {e}")
        return False

# Function to fetch stored PDFs
def get_stored_pdfs():
    try:
        response = index.describe_index_stats()
        return response["total_vector_count"]
    except Exception as e:
        st.error(f"Error fetching stored PDFs: {e}")
        return 0

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
        if pdf_text:
            success = store_pdf_in_pinecone(uploaded_file.name, pdf_text)
            if success:
                st.success(f"✅ {uploaded_file.name} stored successfully!")

# Stored PDFs Section
elif pdf_source == "Choose from the Document Storage":
    st.subheader("📄 Stored Legal Documents")
    total_pdfs = get_stored_pdfs()
    st.write(f"📄 **Total PDFs Stored:** {total_pdfs}" if total_pdfs > 0 else "No PDFs found in storage.")

# Language Selection
st.subheader("🌍 Choose Input Language")
input_language = st.radio("", ["English", "Arabic"], horizontal=True)

st.subheader("🌍 Choose Response Language")
response_language = st.radio("", ["English", "Arabic"], horizontal=True)

# Search Bar
st.markdown("## 🔍 Ask a question (in English or Arabic)")
query = st.text_input("Enter your legal question:")
if query:
    try:
        query_vector = embedding_model.encode(query).tolist()
        results = index.query(vector=query_vector, top_k=5, include_metadata=True)

        st.markdown("### 📜 Relevant Legal Documents:")
        if results and "matches" in results:
            for match in results["matches"]:
                st.write(f"📄 {match['metadata']['filename']} (Score: {match['score']:.2f})")
        else:
            st.warning("No relevant documents found.")
    except Exception as e:
        st.error(f"Error querying Pinecone: {e}")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>Developed with ❤️ using Streamlit & Pinecone</p>", unsafe_allow_html=True)
