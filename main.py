import os
import streamlit as st
import pinecone
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

# Streamlit UI
st.title("📜 Legal HelpDesk for Saudi Arabia")

# PDF Source Selection
st.subheader("Select PDF Source")
pdf_option = st.radio("Choose a source:", ["Upload from PC", "Choose from the Document Storage"])

# File upload
uploaded_pdf = None
if pdf_option == "Upload from PC":
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

# Preloaded PDFs
preloaded_pdfs = ["Law of the Council of Ministers.pdf"]
selected_pdf = None
if pdf_option == "Choose from the Document Storage":
    selected_pdf = st.selectbox("Select a PDF", preloaded_pdfs)

# Language Selection
st.subheader("Choose Input Language")
input_lang = st.radio("Input Language:", ["English", "Arabic"])

st.subheader("Choose Response Language")
response_lang = st.radio("Response Language:", ["English", "Arabic"])

# User query input
st.subheader("Ask a question (in English or Arabic):")
query = st.text_input("Enter your legal query:")

# Process Query
if query:
    try:
        # Generate query vector
        query_vector = embedding_model.encode(query).tolist()

        # Search in Pinecone
        results = index.query(vector=query_vector, top_k=1, include_metadata=True)  # Get only top 1 match

        # Display only the article text
        if results["matches"]:
            best_match = results["matches"][0]  # Get the top-ranked match
            article_text = best_match["metadata"].get("text", "No content available.")
            
            st.subheader("📖 Relevant Legal Article:")
            st.write(article_text)
        else:
            st.info("No relevant article found.")

    except Exception as e:
        st.error(f"Error retrieving results: {e}")

