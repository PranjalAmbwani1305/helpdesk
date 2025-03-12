import os
import uuid
import streamlit as st
import pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") 
INDEX_NAME = "helpdesk"

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Load Hugging Face model for embeddings (384-dimension)
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

st.set_page_config(page_title="AI-Powered Legal HelpDesk", layout="wide")

# Sidebar UI for PDF Selection
st.sidebar.header("📂 Stored PDFs")

# Function to get stored PDFs from Pinecone
def get_stored_pdfs():
    """Fetch unique PDF names stored in Pinecone."""
    stats = index.describe_index_stats()
    vector_count = stats["namespaces"][""]["vector_count"]
    
    # Fetch stored vectors
    results = index.query(vector=[0] * 384, top_k=min(vector_count, 10), include_metadata=True)

    # Extract unique PDF names
    return list(set(match["metadata"]["pdf_name"] for match in results["matches"] if "pdf_name" in match["metadata"]))

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

# Function to store vectors in Pinecone
def store_vectors(embeddings, pdf_name):
    """Store embeddings in Pinecone with unique keys."""
    for idx, embedding in enumerate(embeddings):
        index.upsert(vectors=[
            (f"{pdf_name}_{idx}_{uuid.uuid4().hex[:8]}", embedding, {"pdf_name": pdf_name})
        ])

# Upload and Process PDF
st.header("📜 AI-Powered Legal HelpDesk")
st.subheader("Select PDF Source")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    pdf_name = uploaded_file.name
    pdf_text = extract_text_from_pdf(uploaded_file)

    # Split text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(pdf_text)

    # Generate embeddings
    embeddings = embed_model.embed_documents(chunks)

    # Store vectors in Pinecone
    store_vectors(embeddings, pdf_name)

    st.success(f"✅ PDF '{pdf_name}' uploaded and processed successfully!")

    # Refresh the dropdown with new PDFs
    st.sidebar.selectbox("Select a PDF", options=get_stored_pdfs(), key="pdf_dropdown")

# Display Available PDFs
st.sidebar.selectbox("Select a PDF", options=get_stored_pdfs(), key="pdf_dropdown")

# Question Input
st.subheader("Ask a legal question:")
query = st.text_input("Type your question here...")

if query:
    # Search in Pinecone
    query_embedding = embed_model.embed_query(query)
    search_results = index.query(vector=query_embedding, top_k=5, include_metadata=True)

    st.subheader("📖 Relevant Legal Sections:")
    for match in search_results["matches"]:
        st.write(f"🔹 **From PDF:** {match['metadata']['pdf_name']}")
        st.write(match["metadata"].get("text", "No text available"))
        st.write("---")
