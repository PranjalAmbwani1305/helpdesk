import streamlit as st
import pinecone
import os
import pdfplumber
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Pinecone Setup
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index_name = "helpdesk"

# Check if index exists, otherwise create one
if index_name not in pc.list_indexes().names():
    pc.create_index(name=index_name, dimension=384, metric="cosine")

index = pc.Index(index_name)

# Hugging Face Sentence Embedding Model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def process_pdf(pdf_path, chunk_size=500):
    """Extracts and chunks text from a PDF file."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def store_vectors(chunks, pdf_name):
    """Embeds and stores document chunks in Pinecone."""
    vectors = embedding_model.encode(chunks).tolist()
    upserts = [(f"{pdf_name}-doc-{i}", vectors[i], {"text": chunks[i], "pdf_name": pdf_name}) for i in range(len(chunks))]
    index.upsert(upserts)

def check_existing_pdfs():
    """Retrieves all unique PDF names from Pinecone."""
    existing_pdfs = set()
    results = index.describe_index_stats()

    if results.get("total_vector_count", 0) > 0:
        vector_data = index.query(vector=[0] * 384, top_k=10000, include_metadata=True)
        for match in vector_data.get("matches", []):
            if "pdf_name" in match.get("metadata", {}):
                existing_pdfs.add(match["metadata"]["pdf_name"])
    
    return list(existing_pdfs)

def query_vectors(query, selected_pdf=None):
    """Searches Pinecone for relevant text chunks from a selected or all PDFs."""
    query_vector = embedding_model.encode([query]).tolist()[0]
    filter_condition = {"pdf_name": {"$eq": selected_pdf}} if selected_pdf and selected_pdf != "All PDFs" else None
    results = index.query(vector=query_vector, top_k=8, include_metadata=True, filter=filter_condition)

    if results and "matches" in results:
        matched_texts = [match["metadata"]["text"] for match in results["matches"]]
        return "\n\n".join(matched_texts).strip() if matched_texts else "No relevant information found."
    
    return "No relevant information found."

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>📂 AI-Powered HelpDesk</h1>", unsafe_allow_html=True)

# Select PDF Source
pdf_source = st.radio("📑 Select PDF Source:", ["Upload from PC", "Choose from the Document Storage"])

if pdf_source == "Upload from PC":
    uploaded_file = st.file_uploader("📂 Upload a PDF", type=["pdf"])
    if uploaded_file:
        pdf_name = uploaded_file.name
        existing_pdfs = check_existing_pdfs()

        if pdf_name in existing_pdfs:
            st.warning(f"⚠️ '{pdf_name}' is already stored in Pinecone.")
        else:
            temp_pdf_path = f"temp_{pdf_name}"
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.read())

            chunks = process_pdf(temp_pdf_path)
            store_vectors(chunks, pdf_name)
            st.success("✅ PDF uploaded and stored in Pinecone successfully!")

elif pdf_source == "Choose from the Document Storage":
    existing_pdfs = check_existing_pdfs()
    if existing_pdfs:
        selected_pdf = st.selectbox("📜 Select a stored PDF:", ["All PDFs"] + existing_pdfs)
    else:
        st.warning("⚠️ No PDFs available in the repository. Please upload one.")

# Query Input
query = st.text_input("💬 Ask a question:")

if st.button("Get Answer"):
    if query and (pdf_source == "Choose from the Document Storage" and selected_pdf):
        response = query_vectors(query, selected_pdf)
        st.write(f"**Answer:** {response}")
    else:
        st.warning("⚠️ Please enter a question and select a PDF.")
