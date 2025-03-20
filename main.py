import os
import pinecone
import streamlit as st
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

# Streamlit Page Config (must be first)
st.set_page_config(page_title="Legal HelpDesk", page_icon="⚖️", layout="wide")

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
    return text.strip() if text else "No text extracted from the PDF."

# Function to store PDF content in Pinecone
def store_pdf_in_pinecone(pdf_name, pdf_text):
    try:
        vector = embedding_model.encode(pdf_text).tolist()
        index.upsert(
            vectors=[
                (pdf_name, vector, {"filename": pdf_name})
            ]
        )
        return True
    except Exception as e:
        st.error(f"❌ Error storing PDF in Pinecone: {e}")
        return False

# Function to fetch stored PDFs count
def get_stored_pdfs():
    try:
        response = index.describe_index_stats()
        return response.get("total_vector_count", 0)
    except Exception as e:
        st.error(f"❌ Error fetching stored PDFs: {e}")
        return 0

# UI Header
st.title("⚖️ AI-Powered Legal HelpDesk for Saudi Arabia")
st.write("Helping you find legal information from Saudi Arabian laws quickly and accurately.")

# PDF Source Selection
st.subheader("📂 Select PDF Source")
pdf_source = st.radio("Choose PDF Source:", ["Upload from PC", "Choose from the Document Storage"])

# File Upload Section
if pdf_source == "Upload from PC":
    uploaded_file = st.file_uploader("📄 Upload a PDF", type=["pdf"])
    if uploaded_file:
        pdf_text = extract_text_from_pdf(uploaded_file)
        if pdf_text and pdf_text != "No text extracted from the PDF.":
            success = store_pdf_in_pinecone(uploaded_file.name, pdf_text)
            if success:
                st.success(f"✅ {uploaded_file.name} stored successfully in Pinecone!")
        else:
            st.warning("⚠️ No valid text found in the uploaded PDF.")

# Stored PDFs Section
elif pdf_source == "Choose from the Document Storage":
    st.subheader("📁 Stored Legal Documents")
    total_pdfs = get_stored_pdfs()
    st.write(f"📑 **Total PDFs Stored:** {total_pdfs}")

# Language Selection
st.subheader("🌐 Choose Input Language")
input_language = st.radio("Select input language:", ["English", "Arabic"], horizontal=True)

st.subheader("🌍 Choose Response Language")
response_language = st.radio("Select response language:", ["English", "Arabic"], horizontal=True)

# Search Bar
st.subheader("🔍 Ask a legal question")
query = st.text_input("Enter your legal question:")

if query:
    try:
        query_vector = embedding_model.encode(query).tolist()
        results = index.query(vector=query_vector, top_k=5, include_metadata=True)

        if "matches" in results and results["matches"]:
            st.subheader("📜 Relevant Legal Documents:")
            for match in results["matches"]:
                st.write(f"📄 {match['metadata']['filename']} (Score: {match['score']:.2f})")
        else:
            st.warning("⚠️ No relevant documents found.")

    except Exception as e:
        st.error(f"❌ Error retrieving results: {e}")
