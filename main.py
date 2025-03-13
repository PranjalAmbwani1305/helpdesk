import streamlit as st
import os
from dotenv import load_dotenv
from deep_translator import GoogleTranslator  
from sentence_transformers import SentenceTransformer
import PyPDF2
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Initialize Hugging Face Model for Embeddings
hf_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Pinecone Connection Setup
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "helpdesk"

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if the index exists before using it
if INDEX_NAME in [i.name for i in pc.list_indexes()]:
    index = pc.Index(INDEX_NAME)
else:
    st.error(f"❌ Pinecone index '{INDEX_NAME}' not found! Please create it first.")
    st.stop()

# Function to Process PDF and Extract Text in Chunks
def process_pdf(pdf_path, chunk_size=500):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
    
    text = text.replace("\n", " ")  # Fix newlines that break sentences
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    return chunks

# Check if PDF is Already Stored in Pinecone
def pdf_already_stored(pdf_name):
    try:
        query_results = index.query(vector=[0]*384, top_k=100, include_metadata=True)
        stored_pdfs = set(match["metadata"].get("pdf_name", "") for match in query_results["matches"])
        return pdf_name in stored_pdfs
    except Exception as e:
        st.error(f"❌ Error querying Pinecone: {e}")
        return False

# Store Vectors in Pinecone with Metadata
def store_vectors(chunks, pdf_name):
    vectors = []
    
    for i, chunk in enumerate(chunks):
        embedding = hf_model.encode(chunk).tolist()
        vector_id = f"{pdf_name}-chunk-{i+1}"  # Unique ID per chunk
        
        metadata = {
            "pdf_name": pdf_name,
            "text": chunk.strip(),
            "chunk_id": i+1
        }
        
        vectors.append((vector_id, embedding, metadata))
    
    if vectors:
        index.upsert(vectors)
        st.success(f"✅ {len(vectors)} chunks stored successfully in Pinecone.")

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>AI-Powered Legal HelpDesk</h1>", unsafe_allow_html=True)

# Sidebar: PDF Selection
st.sidebar.header("📂 Select PDF Source")

# PDF Upload
pdf_source = st.radio("Select PDF Source", ["Upload from PC"])

if pdf_source == "Upload from PC":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        temp_pdf_path = f"temp_{uploaded_file.name}"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        if not pdf_already_stored(uploaded_file.name):
            chunks = process_pdf(temp_pdf_path)
            store_vectors(chunks, uploaded_file.name)
            st.success("✅ PDF uploaded and processed!")
        else:
            st.info("ℹ️ This PDF has already been processed!")

st.stop()
