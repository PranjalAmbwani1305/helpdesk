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

if not PINECONE_API_KEY:
    st.error("⚠️ Pinecone API key is missing. Set PINECONE_API_KEY in your environment.")
    st.stop()

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Load Hugging Face model for embeddings
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

st.set_page_config(page_title="AI-Powered Legal HelpDesk", layout="wide")

# Function to get stored PDFs from Pinecone
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
        st.error(f"⚠️ Error fetching PDFs: {e}")
        return []

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

# Function to store vectors in Pinecone
def store_vectors(embeddings, pdf_name):
    """Store embeddings in Pinecone with unique keys."""
    for idx, embedding in enumerate(embeddings):
        unique_id = f"{pdf_name}_{idx}_{uuid.uuid4().hex[:8]}"  # Unique ID for each vector
        index.upsert(vectors=[
            (unique_id, embedding, {"pdf_name": pdf_name})
        ])

# Sidebar - Stored PDFs
st.sidebar.header("📂 Stored PDFs")
stored_pdfs = get_stored_pdfs()
if not stored_pdfs:
    stored_pdfs = ["No PDFs Found"]
selected_pdf = st.sidebar.selectbox("Select a PDF", options=stored_pdfs, key="pdf_dropdown")

# Upload and Process PDF
st.header("📜 AI-Powered Legal HelpDesk")
st.subheader("Upload and Process PDF")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
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
