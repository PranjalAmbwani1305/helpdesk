import os
import pinecone
import streamlit as st
import re
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

# Load environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "helpdesk"

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Load SentenceTransformer model
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
        index.upsert(vectors=[{"id": pdf_name, "values": vector, "metadata": {"pdf_name": pdf_name}}])
        return True
    except Exception as e:
        st.error(f"Error storing PDF in Pinecone: {e}")
        return False

# Function to fetch stored PDFs from Pinecone
def get_stored_pdfs():
    try:
        response = index.describe_index_stats()
        total_vectors = response.get("total_vector_count", 0)

        if total_vectors == 0:
            return []

        query_results = index.query(vector=[0] * 384, top_k=total_vectors, include_metadata=True)
        pdf_names = list(set(match["metadata"].get("pdf_name", "Unknown PDF") for match in query_results["matches"]))

        return pdf_names
    except Exception as e:
        st.error(f"Error fetching stored PDFs: {e}")
        return []

# Function to get stored articles from a selected PDF
def get_stored_articles(selected_pdf):
    try:
        query_results = index.query(vector=[0] * 384, top_k=50, include_metadata=True)
        articles = [
            match["metadata"]
            for match in query_results["matches"]
            if match["metadata"].get("pdf_name") == selected_pdf
        ]
        return articles
    except Exception as e:
        st.error(f"Error fetching articles: {e}")
        return []

# Streamlit UI Layout
st.set_page_config(page_title="Legal HelpDesk", page_icon="⚖️", layout="wide")

# Header
st.title("AI-Powered Legal HelpDesk for Saudi Arabia")
st.write("Helping you find legal information from Saudi Arabian laws quickly and accurately.")

# PDF Source Selection
st.subheader("📂 Select PDF Source")
pdf_source = st.radio("Choose PDF Source:", ["Upload from PC", "Choose from the Document Storage"])

# File Upload Section
if pdf_source == "Upload from PC":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        pdf_text = extract_text_from_pdf(uploaded_file)
        success = store_pdf_in_pinecone(uploaded_file.name, pdf_text)
        if success:
            st.success(f"{uploaded_file.name} stored successfully!")

# Stored PDFs Selection
elif pdf_source == "Choose from the Document Storage":
    st.subheader("📚 Select a PDF from Pinecone")
    pdf_names = get_stored_pdfs()

    if pdf_names:
        selected_pdf = st.selectbox("Select a PDF", pdf_names)
    else:
        st.info("No PDFs found.")

# Choose Input & Response Language
st.subheader("🗣️ Choose Input & Response Language")
col1, col2 = st.columns(2)

with col1:
    input_language = st.radio("Select input language:", ["English", "Arabic"], horizontal=True)

with col2:
    response_language = st.radio("Select response language:", ["English", "Arabic"], horizontal=True)

# Fetch and Display Articles for Selected PDF
if pdf_source == "Choose from the Document Storage" and selected_pdf:
    st.subheader(f"📑 Available Articles in {selected_pdf}")

    articles = get_stored_articles(selected_pdf)
    if articles:
        selected_article = st.selectbox(
            "Select an Article",
            [f"Article {a['article_number']} - {a.get('chapter_number', 'Unknown Chapter')}" for a in articles]
        )
    else:
        st.info("No articles found.")

# Question Input
st.subheader("🔍 Ask a Question")
query = st.text_input("Enter your legal question:")
if query:
    try:
        query_vector = embedding_model.encode(query).tolist()
        results = index.query(vector=query_vector, top_k=5, include_metadata=True)

        st.subheader("📖 Relevant Legal Articles:")
        for match in results["matches"]:
            metadata = match["metadata"]
            article_text = metadata.get("text", "No text available")
            pdf_name = metadata.get("pdf_name", "Unknown PDF")

            st.write(f"📑 **{metadata.get('article_number', 'Unknown Article')}** from **{pdf_name}**")
            st.write(f"✍ {article_text}")
            st.write("———")

    except Exception as e:
        st.error(f"Error retrieving results: {e}")
