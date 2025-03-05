import streamlit as st
import pinecone
import PyPDF2
import numpy as np
import os
import re
import time
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator

# === 📌 Initialize Pinecone === #
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets.get("PINECONE_ENV", "us-east-1")

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index_name = "helpdesk"


time.sleep(5)
index = pc.Index(index_name)
print("✅ Pinecone Index Ready:", index.describe_index_stats())

# === 📌 AI Model === #
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# === 📌 Helper Functions === #
def get_existing_pdfs():
    """Retrieve stored PDFs from Pinecone."""
    existing_pdfs = set()
    try:
        results = index.query(vector=[0]*1536, top_k=1000, include_metadata=True)
        for match in results["matches"]:
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
    
    for title, content in structured_data.items():
        vector = embedder.encode(content).tolist()
        metadata = {"pdf_name": pdf_name, "chapter": title, "text": content}
        index.upsert([(f"{pdf_name}-{title}", vector, metadata)])

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

def find_most_relevant_section(query, sections):
    """Find the most relevant section for a query."""
    try:
        query_embedding = embedder.encode(query)
        section_scores = []

        for section in sections:
            full_text = f"{section['title']} {section['content']}"
            section_embedding = embedder.encode(full_text)

            similarity = np.dot(query_embedding, section_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(section_embedding)
            )

            section_scores.append({'section': section, 'score': similarity})

        if not section_scores:
            return "⚠️ No relevant sections found in the document."

        top_section = max(section_scores, key=lambda x: x['score'])
        return f"**📌 Section:** {top_section['section']['title']}\n\n{top_section['section']['content'][:1000]}..."
    
    except Exception as e:
        st.error(f"Relevance search error: {e}")
        return "Unable to process the query."

def translate_response(response, target_language):
    """Translate response if needed."""
    if target_language.lower() == 'arabic':
        try:
            return GoogleTranslator(source='auto', target='ar').translate(response)
        except Exception as e:
            st.error(f"Translation error: {e}")
            return response
    return response

# === 📌 Streamlit UI === #
def main():
    st.set_page_config(page_title="Saudi Legal HelpDesk", page_icon="⚖️")

    st.title("🏛️ AI-Powered Legal HelpDesk for Saudi Arabia")
    
    # Initialize document storage
    storage_dir = "document_storage"
    os.makedirs(storage_dir, exist_ok=True)

    # Choose action
    action = st.radio("Choose an action:", ["Use existing PDFs", "Upload a new PDF"])

    if action == "Upload a new PDF":
        uploaded_file = st.file_uploader("📂 Upload a PDF", type=["pdf"])
        if uploaded_file:
            file_path = os.path.join(storage_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"✅ Document '{uploaded_file.name}' saved successfully!")

            # Extract and store in Pinecone
