import streamlit as st
import pinecone
import PyPDF2
import os
import re
import time
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer

# Read API Key from Streamlit Secrets
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets.get("PINECONE_ENV", "us-east-1")  # Default to us-east-1

# Initialize Pinecone (Using Your Preferred Method)
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index_name = "helpdesk"

index = pc.Index(index_name)
print("✅ Pinecone Index Ready:", index.describe_index_stats())

# Load Sentence Transformer model for embeddings
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Function to extract structured chapters from PDF
def process_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

    print("📌 Extracted PDF Text (Preview):", text[:500])  # Show first 500 characters

    # Improved Regex: Supports "CHAPTER X", "ARTICLE X", and other legal formats
    chapters = re.split(r'(CHAPTER\s+\d+|ARTICLE\s+\d+)', text, flags=re.IGNORECASE)
    structured_data = {}

    for i in range(1, len(chapters), 2):
        chapter_title = chapters[i].strip()
        chapter_content = chapters[i + 1].strip() if i + 1 < len(chapters) else ""
        
        print(f"📌 Extracted: {chapter_title} -> {len(chapter_content)} characters")  # Debugging

        structured_data[chapter_title] = chapter_content

    return structured_data

# Function to store extracted chapters in Pinecone
def store_vectors(structured_data, pdf_name):
    for title, content in structured_data.items():
        vector = embedder.encode(content).tolist()

        metadata = {
            "pdf_name": pdf_name,
            "chapter": title,  
            "text": content
        }

        print(f"📌 Storing: {title} in Pinecone with {len(vector)} dimensions")
        index.upsert([(f"{pdf_name}-{title}", vector, metadata)])

# Function to check if Pinecone is storing data properly
def debug_pinecone_storage():
    print("📌 Checking Pinecone stored data...")
    
    try:
        index_stats = index.describe_index_stats()
        print("📌 Index Stats:", index_stats)

        if index_stats["total_vector_count"] == 0:
            print("⚠️ No data found in the index. Ensure data is stored first.")
            return

        results = index.query(
            vector=embedder.encode("test query").tolist(),  # Use a real vector
            top_k=5,
            include_metadata=True
        )

        print("📌 Sample stored data:", results)
    except Exception as e:
        print("⚠️ Pinecone Query Failed:", str(e))

# Function to query Pinecone and retrieve the exact chapter
def query_vectors(query, selected_pdf):
    match = re.search(r'(CHAPTER|ARTICLE)\s+(\d+)', query, re.IGNORECASE)
    requested_section = f"{match.group(1).upper()} {match.group(2)}" if match else None

    print(f"🔍 Requested Section: {requested_section}")  # Debugging

    vector = embedder.encode(query).tolist()
    
    results = index.query(
        vector=vector, 
        top_k=5, 
        include_metadata=True, 
        filter={"pdf_name": selected_pdf}
    )

    print("📌 Pinecone Query Results:", results)

    if not results["matches"]:
        return "⚠️ No relevant information found in the selected document."

    for match in results["matches"]:
        stored_chapter = match["metadata"].get("chapter", "")
        stored_text = match["metadata"].get("text", "")

        print(f"📌 Found stored chapter: {stored_chapter}")  # Debugging
        print(f"📌 Stored text preview: {stored_text[:200]}")  # Show first 200 characters

        if requested_section and requested_section in stored_chapter:
            return f"**Extracted Answer from {requested_section}:**\n\n{stored_text}"

    return "⚠️ Requested section not found in the document."

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>📜 AI-Powered Legal HelpDesk</h1>", unsafe_allow_html=True)

# File Uploading Section
uploaded_file = st.file_uploader("📂 Upload a PDF", type=["pdf"])
if uploaded_file:
    temp_pdf_path = f"temp_{uploaded_file.name}"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.read())
    
    structured_data = process_pdf(temp_pdf_path)
    store_vectors(structured_data, uploaded_file.name)
    st.success("✅ PDF uploaded and processed!")

    # Debugging: Check what was stored
    debug_pinecone_storage()

# Select from Uploaded PDFs
pdf_list = [uploaded_file.name] if uploaded_file else []
selected_pdf = st.selectbox("📖 Select PDF for Query", pdf_list) if pdf_list else None

# Language selection
input_lang = st.radio("🌍 Choose Input Language", ["English", "Arabic"], index=0)
response_lang = st.radio("🌍 Choose Response Language", ["English", "Arabic"], index=0)

# User query input
query = st.text_input("🔎 Ask a question (e.g., 'Chapter 5'):" if input_lang == "English" else "📝 اسأل سؤالاً (مثل 'الفصل 5'): ")

if st.button("🔍 Get Answer"):
    if selected_pdf and query:
        # Translate query to English for processing
        detected_lang = GoogleTranslator(source="auto", target="en").translate(query)
        response = query_vectors(detected_lang, selected_pdf)

        # Translate response if needed
        if response_lang == "Arabic":
            response = GoogleTranslator(source="en", target="ar").translate(response)
            st.markdown(f"<div dir='rtl' style='text-align: right;'>{response}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='white-space: pre-wrap; font-family: Arial;'>{response}</div>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a query and select a PDF.")
