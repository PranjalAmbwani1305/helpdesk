import asyncio

# Fix for "RuntimeError: no running event loop"
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.run(asyncio.sleep(0))

import streamlit as st
import pinecone
import PyPDF2
import numpy as np
import os
import re
import time
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator

# === Initialize Pinecone === #
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets.get("PINECONE_ENV", "us-east-1")

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index_name = "helpdesk"

index = pc.Index(index_name)
print("✅ Pinecone Index Ready:", index.describe_index_stats())

# === AI Model === #
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

# === Helper Functions === #
def get_existing_pdfs():
    """Retrieve stored PDFs from Pinecone."""
    existing_pdfs = set()
    try:
        stats = index.describe_index_stats()
        if stats.get("total_vector_count", 0) == 0:
            print("⚠️ No vectors found in Pinecone. The index might be empty.")
            return existing_pdfs

        results = index.query(vector=[0]*1536, top_k=1000, include_metadata=True)
        for match in results.get("matches", []):
            pdf_name = match["metadata"].get("pdf_name", "")
            if pdf_name:
                existing_pdfs.add(pdf_name)
    except Exception as e:
        print("⚠️ Error checking existing PDFs:", e)
    return existing_pdfs


def store_vectors(structured_data, pdf_name):
    """Store extracted document sections into Pinecone."""
    existing_pdfs = get_existing_pdfs()

    if pdf_name in existing_pdfs:
        print(f"⚠️ {pdf_name} already exists in Pinecone. Skipping storage.")
        return

    upsert_data = []
    for section in structured_data:
        title = section["title"]
        content = section["content"]
        vector = embedder.encode(content).tolist()

        metadata = {"pdf_name": pdf_name, "chapter": title, "text": content}
        upsert_data.append((f"{pdf_name}-{title}", vector, metadata))

    if upsert_data:
        index.upsert(upsert_data)
        print(f"✅ Stored {len(upsert_data)} sections in Pinecone for '{pdf_name}'.")
    else:
        print(f"⚠️ No valid sections found in '{pdf_name}', nothing was stored.")


def extract_text(file_path):
    """Extract text from PDF."""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ' '.join([
                page.extract_text() for page in reader.pages if page.extract_text()
            ])
        return preprocess_text(text)
    except Exception as e:
        st.error(f"Error extracting PDF text: {e}")
        return ""


def preprocess_text(text):
    """Clean and format text."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\b\d+\b', '', text)
    return text.strip()


def split_into_sections(text):
    """Split text into structured sections."""
    sections = re.split(r'(Article\s+\d+[:.]|Chapter\s+\d+[:.])', text, flags=re.IGNORECASE)
    
    if len(sections) < 2:
        return [{"title": "Full Document", "content": text}]
    
    cleaned_sections = []
    for i in range(1, len(sections), 2):
        if i+1 < len(sections):
            section = {'title': sections[i].strip(), 'content': sections[i+1].strip()}
            cleaned_sections.append(section)
    
    return cleaned_sections

# === Streamlit UI === #
def main():
    st.set_page_config(page_title="Saudi Legal HelpDesk", page_icon="⚖️")

    st.title("🏛️ AI-Powered Legal HelpDesk for Saudi Arabia")

    # Choose action
    action = st.radio("Choose an action:", ["Use existing PDFs", "Upload a new PDF"])

    if action == "Upload a new PDF":
        uploaded_file = st.file_uploader("📂 Upload a PDF", type=["pdf"])
        if uploaded_file:
            file_path = os.path.join("document_storage", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"✅ Document '{uploaded_file.name}' saved successfully!")

            # Extract and store in Pinecone
            full_text = extract_text(file_path)
            structured_data = split_into_sections(full_text)
            store_vectors(structured_data, uploaded_file.name)

    # Retrieve available PDFs from Pinecone
    existing_pdfs = get_existing_pdfs()

    if existing_pdfs:
        selected_pdf = st.selectbox("📖 Select PDF for Query", list(existing_pdfs), index=0)
    else:
        selected_pdf = None
        st.warning("⚠️ No PDFs found in Pinecone. Please upload a PDF first.")

    # User query input
    query = st.text_input("🔎 Ask a legal question:")

    if st.button("🔍 Get Answer"):
        if not selected_pdf:
            st.warning("⚠️ Please select a PDF before asking a question.")
            return
        
        if not query.strip():
            st.warning("⚠️ Please enter a valid query.")
            return

        # Retrieve text from Pinecone
        results = index.query(
            vector=embedder.encode(query).tolist(), 
            top_k=5, 
            include_metadata=True, 
            filter={"pdf_name": {"$eq": selected_pdf}}
        )

        if results["matches"]:
            response = results["matches"][0]["metadata"]["text"]
        else:
            response = "⚠️ No relevant information found in the selected document."

        st.write(f"**Answer:** {response}")

if __name__ == "__main__":
    main()
