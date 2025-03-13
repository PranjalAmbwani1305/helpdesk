import streamlit as st
import pinecone
import os
import re
from dotenv import load_dotenv
from deep_translator import GoogleTranslator  
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF for better PDF extraction

# Load environment variables
load_dotenv()

# Initialize Hugging Face Model for Embeddings
hf_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Pinecone Connection
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

from pinecone import Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "helpdesk"
index = pc.Index(index_name)

# Function to Process PDF and Extract Cleaned Text
def process_pdf(pdf_path, chunk_size=500):
    doc = fitz.open(pdf_path)
    full_text = []
    
    for page in doc:
        text = page.get_text("text")
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        if text:
            full_text.append(text)
    
    # Join all pages into one document
    full_text = " ".join(full_text)
    
    # Split into meaningful chunks (preserve sentence boundaries)
    sentences = re.split(r'(?<=[.!?])\s+', full_text)
    chunks = []
    temp_chunk = ""
    
    for sentence in sentences:
        if len(temp_chunk) + len(sentence) < chunk_size:
            temp_chunk += " " + sentence
        else:
            chunks.append(temp_chunk.strip())
            temp_chunk = sentence
    
    if temp_chunk:
        chunks.append(temp_chunk.strip())

    return chunks

# Check if PDF is Already Stored in Pinecone
def pdf_already_stored(pdf_name):
    query_results = index.query(vector=[0]*384, top_k=100, include_metadata=True)
    
    stored_pdfs = set(match["metadata"].get("pdf_name", "") for match in query_results["matches"])
    
    return pdf_name in stored_pdfs

# Store Vectors in Pinecone with Metadata
def store_vectors(chunks, pdf_name):
    vectors = []
    
    for i, chunk in enumerate(chunks):
        embedding = hf_model.encode(chunk).tolist()
        vector_id = f"{pdf_name}-chunk-{i+1}"  # Unique ID per chunk
        
        metadata = {
            "pdf_name": pdf_name,
            "text": chunk.strip(),  # Store cleaned text
            "chunk_id": i+1
        }
        
        vectors.append((vector_id, embedding, metadata))
    
    if vectors:
        index.upsert(vectors)
        st.success(f"✅ {len(vectors)} text chunks stored successfully in Pinecone.")

# List Stored PDFs from Pinecone
def list_stored_pdfs():
    query_results = index.query(vector=[0]*384, top_k=100, include_metadata=True)
    pdf_names = set(match["metadata"]["pdf_name"] for match in query_results["matches"])
    return list(pdf_names)

# Query Pinecone for Relevant Information
def query_vectors(query, selected_pdf):
    vector = hf_model.encode(query).tolist()
    
    results = index.query(vector=vector, top_k=5, include_metadata=True, filter={"pdf_name": {"$eq": selected_pdf}})

    if results["matches"]:
        matched_texts = [match["metadata"]["text"] for match in results["matches"]]
        combined_text = "\n\n".join(matched_texts)
        return combined_text
    else:
        return "No relevant information found in the selected document."

# Translate Text
def translate_text(text, target_language):
    return GoogleTranslator(source="auto", target=target_language).translate(text)

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>AI-Powered Legal HelpDesk</h1>", unsafe_allow_html=True)

st.sidebar.header("📂 Stored PDFs")
pdf_list = list_stored_pdfs()
if pdf_list:
    with st.sidebar.expander("📜 View Stored PDFs", expanded=False):
        for pdf in pdf_list:
            st.sidebar.write(f"📄 {pdf}")
else:
    st.sidebar.write("No PDFs stored yet. Upload one!")

selected_pdf = None

# PDF Selection
pdf_source = st.radio("Select PDF Source", ["Upload from PC", "Choose from the Document Storage"])

if pdf_source == "Upload from PC":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        temp_pdf_path = f"temp_{uploaded_file.name}"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        if not pdf_already_stored(uploaded_file.name):
            chunks = process_pdf(temp_pdf_path)
            store_vectors(chunks, uploaded_file.name)
            st.success("PDF uploaded and processed!")
        else:
            st.info("This PDF has already been processed!")

        selected_pdf = uploaded_file.name

elif pdf_source == "Choose from the Document Storage":
    if pdf_list:
        selected_pdf = st.selectbox("Select a PDF", pdf_list)
    else:
        st.warning("No PDFs available in the repository. Please upload one!")

# Search Query
query = st.text_input("Ask a question related to the selected document:")

if query and selected_pdf:
    response = query_vectors(query, selected_pdf)
    st.write(response)

# Translation
if query:
    lang = st.selectbox("Translate Response To:", ["English", "Arabic", "French", "Hindi"])
    translated_text = translate_text(response, lang.lower())
    st.write(f"**Translated:** {translated_text}")
