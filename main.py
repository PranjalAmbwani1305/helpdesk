import streamlit as st
import os
import fitz  # PyMuPDF for PDF processing
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone as PineconeVectorDB

# Load environment variables
load_dotenv()

# Pinecone API setup
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "helpdesk"  # Change this if needed

# Initialize Pinecone
st.write("🔗 Connecting to Pinecone...")
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    st.success("✅ Pinecone connected successfully!")
except Exception as e:
    st.error(f"⚠️ Pinecone connection failed: {e}")
    st.stop()

# Sidebar for PDF Selection
st.sidebar.header("📂 Select PDF Source")
pdf_source = st.sidebar.radio("Upload from:", ["PC", "Document Storage"])

uploaded_file = None
if pdf_source == "PC":
    uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
elif pdf_source == "Document Storage":
    stored_pdfs = ["doc1.pdf", "doc2.pdf"]  # Example
    selected_pdf = st.sidebar.selectbox("Choose a stored PDF", stored_pdfs)
    uploaded_file = selected_pdf if selected_pdf else None

# Sidebar for Language Selection
st.sidebar.header("🌍 Choose Language")
input_language = st.sidebar.radio("Choose Input Language:", ["English", "Arabic"])
response_language = st.sidebar.radio("Choose Response Language:", ["English", "Arabic"])

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

# Main section
st.title("📄 PDF Helpdesk")

if uploaded_file:
    st.sidebar.success(f"📌 PDF Loaded: {uploaded_file.name}")

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
        st.stop()

    # Chunking
    chunks = chunk_text(extracted_text)
    st.write(f"🔍 {len(chunks)} chunks extracted!")

    # Show first few chunks
    for i, chunk in enumerate(chunks[:3]):
        st.write(f"Chunk {i+1}: {chunk}")

    # Initialize embeddings
    st.write("🔢 Generating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Store in Pinecone
    st.write("🚀 Storing embeddings in Pinecone...")
    try:
        vector_db = PineconeVectorDB.from_texts(texts=chunks, embedding=embeddings, index_name=INDEX_NAME)
        st.success("✅ PDF stored in Pinecone successfully!")
    except Exception as e:
        st.error(f"⚠️ Error storing in Pinecone: {e}")

# Ask a Question
st.subheader("❓ Ask a Question")
query = st.text_input("Ask a question (in English or Arabic):")

if query:
    st.write("🔍 Searching for relevant information...")
    # Placeholder for LLM or retrieval system
    st.success("🤖 Answer: This is a placeholder response.")
