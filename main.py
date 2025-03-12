import os
import uuid
import streamlit as st
import pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")  # Use env variables for security
INDEX_NAME = "helpdesk"

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Load Hugging Face model for embeddings
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Streamlit UI
st.set_page_config(page_title="AI-Powered Legal HelpDesk for Saudi Arabia", layout="wide")

st.header("📜 AI-Powered Legal HelpDesk for Saudi Arabia")
st.subheader("Select PDF Source")

# Sidebar - Stored PDFs
st.sidebar.header("📂 Stored PDFs")
stored_pdfs = []
try:
    stats = index.describe_index_stats()
    if "namespaces" in stats and "" in stats["namespaces"]:
        vector_count = stats["namespaces"][""]["vector_count"]
        if vector_count > 0:
            results = index.query(vector=[0] * 384, top_k=vector_count, include_metadata=True)
            stored_pdfs = list(set(match["metadata"]["pdf_name"] for match in results["matches"] if "pdf_name" in match["metadata"]))
except Exception as e:
    st.sidebar.error(f"Error fetching PDFs: {e}")

selected_pdf = st.sidebar.selectbox("Select a PDF", options=stored_pdfs if stored_pdfs else ["No PDFs Found"])

# Language Selection
input_lang = st.radio("Choose Input Language", ["English", "Arabic"], index=0)
response_lang = st.radio("Choose Response Language", ["English", "Arabic"], index=0)

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_file):
    try:
        reader = PdfReader(pdf_file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text if text.strip() else None
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

# OCR Function for Scanned PDFs
def extract_text_from_scanned_pdf(pdf_file):
    images = convert_from_path(pdf_file)
    text = "\n".join([pytesseract.image_to_string(Image.open(img)) for img in images])
    return text if text.strip() else None

# Store vectors in Pinecone
def store_vectors(embeddings, chunks, pdf_name):
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        article_id = f"{pdf_name}_article_{idx}"  # Unique article ID
        unique_id = f"{article_id}_{uuid.uuid4().hex[:8]}"  # Unique vector ID
        index.upsert(vectors=[
            (unique_id, embedding, {"pdf_name": pdf_name, "article_id": article_id, "text": chunk})
        ])

# Upload & Process PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    pdf_name = uploaded_file.name
    pdf_text = extract_text_from_pdf(uploaded_file) or extract_text_from_scanned_pdf(uploaded_file)

    if not pdf_text:
        st.error(f"❌ Could not extract text from {pdf_name}. It may be a scanned document.")
    else:
        # Split text into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(pdf_text)

        # Generate embeddings
        embeddings = embed_model.embed_documents(chunks)

        # Store vectors in Pinecone
        store_vectors(embeddings, chunks, pdf_name)

        st.success(f"✅ PDF '{pdf_name}' uploaded and processed successfully!")
        st.sidebar.selectbox("Select a PDF", options=stored_pdfs + [pdf_name], key="pdf_dropdown")

# Question Input
st.subheader("Ask a question (in English or Arabic):")
query = st.text_input("Type your question here...")

if query and selected_pdf != "No PDFs Found":
    query_embedding = embed_model.embed_query(query)
    search_results = index.query(vector=query_embedding, top_k=5, include_metadata=True)

    st.subheader("📖 Relevant Legal Sections:")
    for match in search_results["matches"]:
        st.write(f"🔹 **From PDF:** {match['metadata']['pdf_name']}")
        st.write(f"📜 **Article ID:** {match['metadata'].get('article_id', 'Unknown')}")
        st.write(match["metadata"].get("text", "No text available"))
        st.write("---")
