import os
import fitz  # PyMuPDF for PDF text extraction
import pinecone
import streamlit as st
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "helpdesk"
index = pc.Index(INDEX_NAME)

# Load Sentence Transformer Model
hf_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ------------------------------ PDF Processing ------------------------------
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text_list = []
    pdf_document = fitz.open(pdf_path)
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        text = page.get_text("text")
        text_list.append(text)
    return "\n".join(text_list)

def store_pdf_in_pinecone(pdf_name, text):
    """Splits text, creates embeddings, and stores in Pinecone."""
    paragraphs = text.split("\n\n")
    for i, para in enumerate(paragraphs):
        if len(para.strip()) > 10:  # Skip very short texts
            embedding = hf_model.encode(para).tolist()
            metadata = {
                "pdf_name": pdf_name,
                "text": para,
                "article_number": str(i + 1),
                "type": "article"
            }
            index.upsert([(f"{pdf_name}-article-{i}", embedding, metadata)])

def process_and_store_pdfs(uploaded_files):
    """Processes multiple PDFs, extracts text, and stores in Pinecone."""
    for uploaded_file in uploaded_files:
        pdf_name = uploaded_file.name
        text = extract_text_from_pdf(uploaded_file)
        store_pdf_in_pinecone(pdf_name, text)
        st.success(f"📄 '{pdf_name}' uploaded and processed successfully!")

# ------------------------------ Streamlit UI ------------------------------
st.set_page_config(page_title="Legal HelpDesk", layout="wide")

st.title("📜 AI-Powered Legal HelpDesk for Saudi Arabia")
st.subheader("Upload Legal Documents & Ask Questions")

# File Upload (Multiple PDFs)
uploaded_files = st.file_uploader("Upload PDF Documents", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    process_and_store_pdfs(uploaded_files)

# Language Selection
col1, col2 = st.columns(2)
with col1:
    input_lang = st.radio("Choose Input Language", ["English", "Arabic"])
with col2:
    response_lang = st.radio("Choose Response Language", ["English", "Arabic"])

# Query Box
query = st.text_input("Ask a question (in English or Arabic):", "")

if query:
    query_embedding = hf_model.encode(query).tolist()
    results = index.query(vector=query_embedding, top_k=10, include_metadata=True)

    st.subheader(f"🔍 Showing {len(results['matches'])} results")

    for i, match in enumerate(results["matches"], start=1):
        metadata = match["metadata"]
        st.markdown(f"### {i}")
        st.markdown(f"**📄 Document:** {metadata['pdf_name']}")
        st.markdown(f"**📜 Article:** {metadata['article_number']}")
        st.markdown(f"**✍️ Text:** {metadata['text']}")
        st.markdown("---")
