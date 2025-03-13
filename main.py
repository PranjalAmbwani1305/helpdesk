import streamlit as st
import pinecone
import fitz  # PyMuPDF
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone

# Load environment variables
load_dotenv()

# Pinecone setup
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("PINECONE_INDEX")

# Initialize Pinecone
st.write("🔗 Connecting to Pinecone...")
try:
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    index = pinecone.Index(INDEX_NAME)
    st.write("✅ Pinecone connected successfully!")
except Exception as e:
    st.error(f"⚠️ Pinecone connection failed: {e}")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

# Function to chunk text
def chunk_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_text(text)

# Streamlit UI
st.title("📄 PDF to Pinecone Uploader")

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])
if uploaded_file:
    st.success(f"📂 Uploaded file: {uploaded_file.name}")

    # Save PDF temporarily
    temp_pdf_path = f"temp_{uploaded_file.name}"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    st.write(f"✅ File saved as: {temp_pdf_path}")

    # Extract text
    extracted_text = extract_text_from_pdf(temp_pdf_path)
    if extracted_text:
        st.write("📜 Extracted text successfully!")
    else:
        st.error("⚠️ No text extracted from PDF.")

    # Chunking
    chunks = chunk_text(extracted_text)
    st.write(f"🔍 {len(chunks)} chunks extracted!")

    # Show first few chunks
    for i, chunk in enumerate(chunks[:3]):
        st.write(f"Chunk {i+1}: {chunk}")

    # Initialize embeddings
    st.write("🔢 Generating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Pinecone.from_texts(texts=chunks, embedding=embeddings, index_name=INDEX_NAME)

    st.success("✅ PDF stored in Pinecone successfully!")
