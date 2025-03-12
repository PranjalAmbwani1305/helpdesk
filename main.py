import os
import uuid
import streamlit as st
import pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader

# Load API keys securely from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-west1-gcp")  # Default environment
INDEX_NAME = "helpdesk"

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Load Hugging Face model for embeddings
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Streamlit Page Configuration
st.set_page_config(page_title="AI-Powered Legal HelpDesk", layout="wide")

# Sidebar UI for PDF Selection
st.sidebar.header("📂 Stored PDFs")

# Function to fetch stored PDFs from Pinecone
def get_stored_pdfs():
    """Fetch unique PDF names stored in Pinecone."""
    try:
        stats = index.describe_index_stats()
        if stats.get("total_vector_count", 0) == 0:
            return []

        # Query Pinecone to get stored PDFs
        results = index.query(vector=[0] * 384, top_k=1000, include_metadata=True)
        
        # Extract unique PDF names
        pdf_names = list(set(
            match["metadata"]["pdf_name"] for match in results["matches"] if "pdf_name" in match["metadata"]
        ))

        return pdf_names
    except Exception as e:
        st.error(f"Error fetching PDFs: {e}")
        return []

# Load existing PDFs into sidebar dropdown
stored_pdfs = get_stored_pdfs()
selected_pdf = st.sidebar.selectbox("Select a PDF", options=stored_pdfs if stored_pdfs else ["No PDFs Found"])

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    """Extracts text from an uploaded PDF file."""
    reader = PdfReader(pdf_file)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

# Function to store vectors in Pinecone
def store_vectors(embeddings, pdf_name, chunks):
    """Store embeddings in Pinecone with unique keys."""
    to_upsert = []
    for idx, (embedding, text_chunk) in enumerate(zip(embeddings, chunks)):
        unique_id = f"{pdf_name}_{idx}_{uuid.uuid4().hex[:8]}"  # Unique ID for each vector
        to_upsert.append((unique_id, embedding, {"pdf_name": pdf_name, "text": text_chunk}))

    index.upsert(vectors=to_upsert)

# Upload and Process PDF
st.header("📜 AI-Powered Legal HelpDesk")
st.subheader("Upload a Legal Document")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    pdf_name = uploaded_file.name
    pdf_text = extract_text_from_pdf(uploaded_file)

    if pdf_text.strip():
        # Split text into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(pdf_text)

        # Generate embeddings
        embeddings = embed_model.embed_documents(chunks)

        # Store vectors in Pinecone
        store_vectors(embeddings, pdf_name, chunks)

        st.success(f"✅ PDF '{pdf_name}' uploaded and processed successfully!")

        # Refresh the dropdown with new PDFs
        stored_pdfs = get_stored_pdfs()
        selected_pdf = st.sidebar.selectbox("Select a PDF", options=stored_pdfs)

# Question Input
st.subheader("Ask a legal question:")
query = st.text_input("Type your question here...")

if query and selected_pdf != "No PDFs Found":
    # Search in Pinecone
    query_embedding = embed_model.embed_query(query)
    search_results = index.query(vector=query_embedding, top_k=5, include_metadata=True)

    st.subheader("📖 Relevant Legal Sections:")
    for match in search_results["matches"]:
        st.write(f"🔹 **From PDF:** {match['metadata']['pdf_name']}")
        st.write(match["metadata"].get("text", "No text available"))
        st.write("---")
