import os
import pinecone
import streamlit as st
import tempfile
import fitz  # PyMuPDF for PDF processing
from transformers import AutoTokenizer, AutoModel
import torch

# Load Environment Variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "helpdesk"

# ✅ Initialize Pinecone (New SDK)
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# ✅ Load Hugging Face Embedding Model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Function to Extract Text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Function to Generate Embeddings using Hugging Face Model
def generate_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()  # Mean pooling

# Function to Fetch Stored PDFs from Pinecone
def get_stored_pdfs():
    try:
        response = index.describe_index_stats()
        vector_count = response.total_vector_count  # Correct way to access count

        stored_pdfs = []
        if vector_count > 0:
            vectors = index.query(vector_ids=[], top_k=vector_count, include_metadata=True)
            for vector in vectors.matches:  # Correct attribute
                if "metadata" in vector and "filename" in vector.metadata:
                    stored_pdfs.append(vector.metadata["filename"])
        return stored_pdfs
    except Exception as e:
        st.error(f"Error fetching stored PDFs: {e}")
        return []

# Function to Upload PDF and Store Vectors in Pinecone
def upload_pdf_to_pinecone(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_pdf_path = temp_file.name

        # Extract text from PDF
        pdf_text = extract_text_from_pdf(temp_pdf_path)

        # Generate embeddings for PDF text
        vector = generate_embeddings(pdf_text)

        # Store in Pinecone
        index.upsert(vectors=[
            {"id": uploaded_file.name, "values": vector, "metadata": {"filename": uploaded_file.name}}
        ])

        st.success(f"📄 '{uploaded_file.name}' uploaded successfully!")
    except Exception as e:
        st.error(f"Error uploading PDF: {e}")

# Function to Search Pinecone for Similar Documents
def search_pinecone(query):
    query_vector = generate_embeddings(query)
    results = index.query(vector=query_vector, top_k=5, include_metadata=True)

    response_text = "🔍 **Search Results:**\n"
    for match in results.matches:
        if match.metadata and "filename" in match.metadata:
            response_text += f"- 📄 {match.metadata['filename']} (Score: {match.score:.2f})\n"
    
    return response_text if results.matches else "No relevant documents found."

# Streamlit UI
st.title("⚖️ AI-Powered Legal HelpDesk for Saudi Arabia")

# File Upload
uploaded_file = st.file_uploader("📂 Upload a Legal Document (PDF)", type=["pdf"])
if uploaded_file:
    upload_pdf_to_pinecone(uploaded_file)

# Sidebar: Stored PDFs
st.sidebar.header("📁 Stored Legal Documents")
stored_pdfs = get_stored_pdfs()
if stored_pdfs:
    for pdf in stored_pdfs:
        st.sidebar.write(f"📄 {pdf}")
else:
    st.sidebar.write("No PDFs stored yet.")

# Search Functionality
query = st.text_input("🔍 Enter a legal query:")
if query:
    response = search_pinecone(query)
    st.write(response)
