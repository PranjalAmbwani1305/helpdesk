import os
import uuid
import streamlit as st
import pinecone
import pytesseract
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from PIL import Image

# 🔹 Set up Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "helpdesk"

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# 🔹 Load embedding model
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 🔹 Function to extract text from a text-based PDF
def extract_text_from_pdf(pdf_file):
    """Extracts text from a text-based PDF."""
    reader = PdfReader(pdf_file)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

# 🔹 Function to extract text from a scanned PDF using OCR
def extract_text_from_scanned_pdf(pdf_file):
    """Converts PDF pages to images and extracts text using OCR."""
    images = convert_from_path(pdf_file)
    ocr_text = "\n".join([pytesseract.image_to_string(img) for img in images])
    return ocr_text

# 🔹 Function to store vectors in Pinecone
def store_vectors(embeddings, text_chunks, pdf_name):
    """Store chunked embeddings as articles in Pinecone."""
    upsert_data = []
    
    for idx, (embedding, text) in enumerate(zip(embeddings, text_chunks)):
        article_id = f"{pdf_name}_article_{idx}"  # Unique ID per chunk
        vector_id = f"{article_id}_{uuid.uuid4().hex[:8]}"  # Unique vector ID

        metadata = {
            "pdf_name": pdf_name,
            "article_id": article_id,
            "text": text
        }

        upsert_data.append((vector_id, embedding, metadata))

    if upsert_data:
        try:
            index.upsert(vectors=upsert_data)
            print(f"✅ Successfully stored {len(upsert_data)} chunks from {pdf_name}")
        except Exception as e:
            print(f"❌ Error storing vectors: {e}")

# 🔹 Streamlit UI
st.header("📜 AI-Powered Legal HelpDesk")

uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        pdf_name = uploaded_file.name
        pdf_text = extract_text_from_pdf(uploaded_file)

        if not pdf_text.strip():
            st.warning(f"⚠️ No text found in {pdf_name}. Running OCR...")
            pdf_text = extract_text_from_scanned_pdf(uploaded_file)

        if not pdf_text.strip():
            st.error(f"❌ Could not extract text from {pdf_name}. It may be an image-based PDF.")
            continue

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        text_chunks = text_splitter.split_text(pdf_text)

        embeddings = embed_model.embed_documents(text_chunks)

        store_vectors(embeddings, text_chunks, pdf_name)

        st.success(f"✅ PDF '{pdf_name}' uploaded and processed successfully!")
