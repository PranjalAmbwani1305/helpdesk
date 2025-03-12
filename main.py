import os
import uuid
import streamlit as st
import pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import pytesseract
from pdf2image import convert_from_bytes

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "helpdesk"

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Load Hugging Face model for embeddings
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

st.set_page_config(page_title="AI-Powered Legal HelpDesk", layout="wide")

# Sidebar UI for PDF Selection
st.sidebar.header("📂 Stored PDFs")

def get_stored_pdfs():
    """Fetch unique PDF names stored in Pinecone."""
    try:
        stats = index.describe_index_stats()
        if "namespaces" in stats and "" in stats["namespaces"]:
            vector_count = stats["namespaces"][""]["vector_count"]
            if vector_count == 0:
                return []
            
            # Fetch stored vectors
            results = index.query(vector=[0] * 384, top_k=vector_count, include_metadata=True)

            # Extract unique PDF names
            pdf_names = list(set(
                match["metadata"]["pdf_name"] for match in results["matches"] if "pdf_name" in match["metadata"]
            ))

            return pdf_names
        return []
    except Exception as e:
        st.error(f"Error fetching PDFs: {e}")
        return []

stored_pdfs = get_stored_pdfs()

# Ensure selectbox is used only ONCE
selected_pdf = None
if stored_pdfs:
    selected_pdf = st.sidebar.selectbox("Select a PDF", options=stored_pdfs, key="pdf_dropdown")

# File Upload
st.header("📜 AI-Powered Legal HelpDesk")
st.subheader("Select PDF Source")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    pdf_name = uploaded_file.name
    pdf_text = None

    try:
        reader = PdfReader(uploaded_file)
        pdf_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    except:
        pass

    if not pdf_text:
        st.warning(f"❌ Could not extract text from {pdf_name}. Trying OCR...")
        uploaded_file.seek(0)  # Reset file pointer
        pdf_text = extract_text_with_ocr(uploaded_file)

    if not pdf_text:
        st.error("❌ This document appears to be an image-based PDF and OCR could not extract text.")
    else:
        # Process and store
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(pdf_text)

        embeddings = embed_model.embed_documents(chunks)

        for idx, (embedding, chunk) in enumerate(zip(embeddings, chunks)):
            article_id = f"{pdf_name}-article-{idx+1}"  # Unique Article ID
            index.upsert(vectors=[
                (article_id, embedding, {
                    "pdf_name": pdf_name,
                    "article_id": article_id,
                    "title": f"Article {idx+1}",
                    "text": chunk,
                    "type": "article"
                })
            ])

        st.success(f"✅ PDF '{pdf_name}' uploaded and processed successfully!")

        # Refresh stored PDFs
        stored_pdfs = get_stored_pdfs()

# Question Input
st.subheader("Ask a legal question:")
query = st.text_input("Type your question here...")

if query and selected_pdf:
    query_embedding = embed_model.embed_query(query)
    search_results = index.query(vector=query_embedding, top_k=5, include_metadata=True)

    st.subheader("📖 Relevant Legal Sections:")
    for match in search_results["matches"]:
        pdf_name = match["metadata"].get("pdf_name", "Unknown")
        article_id = match["metadata"].get("article_id", "Unknown")
        title = match["metadata"].get("title", f"Article {article_id.split('-')[-1]}" if "article-" in article_id else "Unknown")
        text = match["metadata"].get("text", "No text available")

        # Display correct Article ID
        st.write(f"🔹 **From PDF:** {pdf_name}")
        st.write(f"📜 **{title}** (ID: `{article_id}`)")
        st.write(text)
        st.write("---")
