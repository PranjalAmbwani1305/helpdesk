import streamlit as st
import pinecone
import pytesseract
from pdf2image import convert_from_bytes
from PyPDF2 import PdfReader
from PIL import Image
import json
import os

# 🔹 Initialize Pinecone
PINECONE_API_KEY = "your-pinecone-api-key"
INDEX_NAME = "your-index-name"

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# 🔹 Function to Extract Text from Scanned PDFs (OCR)
def extract_text_with_ocr(pdf_file):
    """
    Extract text from a scanned (image-based) PDF using OCR.
    :param pdf_file: Uploaded PDF file (BytesIO)
    :return: Extracted text as a string
    """
    text = []
    try:
        images = convert_from_bytes(pdf_file.read(), dpi=300)
        for img in images:
            img = img.convert("L")  # Convert to grayscale for better OCR
            extracted_text = pytesseract.image_to_string(img, lang="eng")
            text.append(extracted_text)
        return "\n".join(text).strip()
    except Exception as e:
        return None  # Return None if OCR fails

# 🔹 Function to Store Data in Pinecone
def store_in_pinecone(pdf_name, text_chunks):
    """
    Stores extracted text chunks in Pinecone with metadata.
    :param pdf_name: Name of the uploaded PDF
    :param text_chunks: List of dictionaries containing chunked text
    """
    vectors = []
    for chunk_id, chunk in enumerate(text_chunks):
        vector_id = f"{pdf_name}-article-{chunk_id}"
        metadata = {
            "pdf_name": pdf_name,
            "article_id": f"article-{chunk_id}",
            "text": chunk["text"],
            "title": chunk.get("title", ""),
            "chapter": chunk.get("chapter", ""),
            "type": "article"
        }
        vectors.append((vector_id, [0.5] * 1536, metadata))  # Dummy vector embedding

    if vectors:
        index.upsert(vectors)
        st.success(f"✅ Successfully stored {len(vectors)} chunks in Pinecone.")

# 🔹 File Upload Section
st.header("📜 AI-Powered Legal HelpDesk")
st.subheader("Select PDF Source")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    pdf_name = uploaded_file.name
    pdf_text = None

    try:
        reader = PdfReader(uploaded_file)
        pdf_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    except:
        pass

    if not pdf_text:
        st.warning(f"❌ Could not extract text from {pdf_name}. Trying OCR...")
        uploaded_file.seek(0)  # Reset file pointer
        pdf_text = extract_text_with_ocr(uploaded_file)

    if not pdf_text:
        st.error("❌ This document appears to be an image-based PDF, and OCR could not extract text.")
    else:
        st.success(f"✅ Text successfully extracted from '{pdf_name}'.")

        # 🔹 Chunk the Text and Store in Pinecone
        text_chunks = [{"text": chunk.strip()} for chunk in pdf_text.split("\n\n") if chunk.strip()]
        store_in_pinecone(pdf_name, text_chunks)

# 🔹 Retrieve Stored PDFs from Pinecone
stored_pdfs = [match["metadata"]["pdf_name"] for match in index.query([], top_k=100, include_metadata=True)["matches"]]
stored_pdfs = list(set(stored_pdfs))  # Remove duplicates

# 🔹 Sidebar: Select a Stored PDF
if stored_pdfs:
    selected_pdf = st.sidebar.selectbox("📂 Stored PDFs", options=stored_pdfs, key="pdf_dropdown")

    if st.sidebar.button("Retrieve Stored Data"):
        query_results = index.query([], top_k=10, include_metadata=True, filter={"pdf_name": selected_pdf})["matches"]
        
        if query_results:
            st.subheader(f"📖 Extracted Articles from '{selected_pdf}'")
            for match in query_results:
                st.markdown(f"**🔹 {match['metadata'].get('title', 'Unknown Title')}**")
                st.markdown(f"📜 **Article ID:** {match['metadata'].get('article_id', 'Unknown')}")
                st.markdown(f"**📖 Text:** {match['metadata'].get('text', '')}")
                st.markdown("---")
        else:
            st.warning("No articles found for this PDF.")

# 🔹 Ask a Question
st.subheader("Ask a legal question:")
question = st.text_input("Type your question here...")

if question and stored_pdfs:
    query_results = index.query([], top_k=5, include_metadata=True)
    
    if query_results:
        st.subheader("📖 Relevant Legal Sections:")
        for match in query_results["matches"]:
            st.markdown(f"🔹 **From PDF:** {match['metadata']['pdf_name']}")
            st.markdown(f"📜 **Article ID:** {match['metadata'].get('article_id', 'Unknown')}")
            st.markdown(f"**{match['metadata'].get('text', '')}**")
            st.markdown("---")
    else:
        st.warning("No relevant sections found.")
