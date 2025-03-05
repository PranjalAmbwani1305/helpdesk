import streamlit as st
import pinecone
import PyPDF2
import os
import pdfplumber
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import re

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
index_name = "helpdesk"

from pinecone import Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(index_name)

model = SentenceTransformer("all-MiniLM-L6-v2")

# Process PDF and extract text
def process_pdf(pdf_path, chunk_size=500):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    
    # Clean up text by removing unnecessary line breaks and excessive spaces
    text = re.sub(r'\n+', ' ', text)  # Remove excessive line breaks
    text = re.sub(r'\s{2,}', ' ', text)  # Remove excessive spaces
    
    print(f"Extracted Text: {text[:1000]}")  # Print first 1000 characters to check
    
    if not text:
        raise ValueError("No text extracted from PDF. Please check the PDF file.")
    
    # Split the text into paragraphs
    paragraphs = text.split("\n")
    print(f"Extracted Paragraphs: {paragraphs[:10]}")  # Print first 10 paragraphs to check

    # Filter out empty paragraphs or irrelevant text
    paragraphs = [para.strip() for para in paragraphs if para.strip()]

    # Chunk paragraphs into manageable pieces
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        para_len = len(para)
        if current_length + para_len > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [para]  # Start new chunk with this paragraph
            current_length = para_len
        else:
            current_chunk.append(para)
            current_length += para_len

    # Add last chunk if there's any remaining content
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Store the extracted chunks in Pinecone
def store_vectors(chunks, pdf_name):
    for i, chunk in enumerate(chunks):
        vector = model.encode(chunk).tolist()
        index.upsert([(f"{pdf_name}-doc-{i}", vector, {"pdf_name": pdf_name, "text": chunk})])

# Query the stored vectors
def query_vectors(query, selected_pdf):
    vector = model.encode(query).tolist()
    results = index.query(vector=vector, top_k=5, include_metadata=True, filter={"pdf_name": {"$eq": selected_pdf}})
    
    if results["matches"]:
        matched_texts = [match["metadata"]["text"] for match in results["matches"]]
        return "\n\n".join(matched_texts)
    else:
        return "No relevant information found in the selected document."

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>AI-Powered Legal HelpDesk</h1>", unsafe_allow_html=True)

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    temp_pdf_path = f"temp_{uploaded_file.name}"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    chunks = process_pdf(temp_pdf_path)
    store_vectors(chunks, uploaded_file.name)
    st.success("PDF uploaded and processed!")

# Query input
query = st.text_input("Ask a legal question:")
if st.button("Get Answer"):
    if uploaded_file and query:
        response = query_vectors(query, uploaded_file.name)
        st.write(f"**Answer:** {response}")
    else:
        st.warning("Please upload a PDF and enter a query.")
