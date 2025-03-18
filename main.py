import streamlit as st
import pinecone
import fitz  # PyMuPDF for PDF text extraction
import hashlib
import os

# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "helpdesk"

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Streamlit UI
st.title("📜 AI-Powered Legal HelpDesk for Saudi Arabia")

# Sidebar: Show uploaded PDFs
st.sidebar.title("📂 Uploaded PDFs")

if "uploaded_pdfs" not in st.session_state:
    st.session_state["uploaded_pdfs"] = {}

# Fetch PDFs from Pinecone
pinecone_docs = set()
query_results = index.query(queries=[[0] * 384], top_k=50, include_metadata=True)  # Dummy query to fetch stored PDFs
if query_results and query_results.get('results'):
    for match in query_results['results'][0]['matches']:
        pdf_name = match['metadata'].get("pdf_name")
        if pdf_name:
            pinecone_docs.add(pdf_name)

# Combine session PDFs and Pinecone PDFs
all_pdfs = set(st.session_state["uploaded_pdfs"].keys()).union(pinecone_docs)

if all_pdfs:
    selected_pdf = st.sidebar.selectbox("Select a PDF:", list(all_pdfs))
    st.session_state["selected_pdf"] = selected_pdf
else:
    st.sidebar.warning("No PDFs uploaded yet.")

# File Upload
st.subheader("Upload a PDF")
uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

if uploaded_file:
    pdf_name = uploaded_file.name
    if pdf_name not in st.session_state["uploaded_pdfs"]:
        st.session_state["uploaded_pdfs"][pdf_name] = uploaded_file
        st.sidebar.success(f"📄 {pdf_name} added!")

# Function: Extract Text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text("text") + "\n"
    return text.strip()

# Store PDF in Pinecone
def store_pdf_in_pinecone(pdf_name, pdf_content):
    pdf_hash = hashlib.md5(pdf_name.encode()).hexdigest()  # Unique ID
    index.upsert([(pdf_hash, [0] * 384, {"pdf_name": pdf_name, "text": pdf_content})])
    st.success(f"✅ {pdf_name} stored in Pinecone.")

# Process PDF and Store
if uploaded_file:
    pdf_text = extract_text_from_pdf(uploaded_file)
    store_pdf_in_pinecone(pdf_name, pdf_text)

# Ask a Question
st.subheader("Ask a Question")
question = st.text_area("Enter your question (English or Arabic):")

if st.button("Search"):
    if "selected_pdf" in st.session_state and st.session_state["selected_pdf"]:
        query_pdf = st.session_state["selected_pdf"]
        query_results = index.query(queries=[[0] * 384], top_k=3, include_metadata=True)
        if query_results and query_results.get('results'):
            st.subheader(f"🔍 Results from {query_pdf}:")
            for match in query_results['results'][0]['matches']:
                st.write(f"📜 {match['metadata']['text']}")
        else:
            st.warning("No relevant results found.")
    else:
        st.warning("Please select a PDF from the sidebar.")

