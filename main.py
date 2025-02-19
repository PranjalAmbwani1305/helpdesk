import streamlit as st
from pinecone import Pinecone
import pdfplumber
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index_name = "helpdesk"

if index_name not in pc.list_indexes():
    pc.create_index(index_name, dimension=768, metric="cosine")
index = pc.Index(index_name)

# Load Hugging Face embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def process_pdf(pdf_path, chunk_size=500):
    """Extracts and chunks text from a PDF."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def store_vectors(chunks, pdf_name):
    """Embeds and stores chunks in Pinecone."""
    vectors = embedding_model.encode(chunks).tolist()
    
    upserts = [(f"{pdf_name}-doc-{i}", vectors[i], {"pdf_name": pdf_name, "text": chunks[i]}) for i in range(len(chunks))]
    index.upsert(upserts)

def query_vectors(query, selected_pdf):
    """Searches for relevant text using Pinecone."""
    query_vector = embedding_model.encode([query]).tolist()[0]
    results = index.query(vector=query_vector, top_k=5, include_metadata=True, filter={"pdf_name": selected_pdf})
    
    if results and "matches" in results:
        matched_texts = [match["metadata"]["text"] for match in results["matches"]]
        return "\n\n".join(matched_texts)
    return "No relevant information found."

# Streamlit UI
st.title("🔍 AI-Powered Legal HelpDesk")

pdf_source = st.radio("Select PDF Source", ["Upload from PC"])
selected_pdf = None

if pdf_source == "Upload from PC":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        temp_pdf_path = f"temp_{uploaded_file.name}"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.read())
        
        chunks = process_pdf(temp_pdf_path)
        store_vectors(chunks, uploaded_file.name)
        st.success("PDF uploaded and processed!")
        selected_pdf = uploaded_file.name

query = st.text_input("Ask a legal question:")
if st.button("Get Answer"):
    if selected_pdf and query:
        with st.spinner("Searching..."):
            response = query_vectors(query, selected_pdf)
        st.write(f"**Answer:** {response}")
    else:
        st.warning("Please upload a PDF and enter a question.")
