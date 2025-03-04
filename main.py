import streamlit as st
import pinecone
import PyPDF2
import os
import re
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# API Keys from .env file
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index_name = "helpdesk"

# Ensure Index Exists
if index_name not in pc.list_indexes().names():
    pc.create_index(name=index_name, dimension=768, metric="cosine")
index = pc.Index(index_name)

# Load Sentence Transformer model for embeddings
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Function to extract structured chapters from PDF
def process_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

    # Improved Regex: Supports "CHAPTER X", "ARTICLE X", and other legal formats
    chapters = re.split(r'(CHAPTER\s+\d+|ARTICLE\s+\d+)', text, flags=re.IGNORECASE)
    structured_data = {}

    for i in range(1, len(chapters), 2):
        chapter_title = chapters[i].strip()
        chapter_content = chapters[i + 1].strip() if i + 1 < len(chapters) else ""
        structured_data[chapter_title] = chapter_content

    return structured_data

# Function to store extracted chapters in Pinecone
def store_vectors(structured_data, pdf_name):
    for title, content in structured_data.items():
        vector = embedder.encode(content).tolist()
        
        # Debugging: Ensure the data is being stored correctly
        print(f"Storing: {title} -> {len(vector)} dimensions")
        
        index.upsert([(f"{pdf_name}-{title}", vector, {"pdf_name": pdf_name, "chapter": title, "text": content})])

# Function to query Pinecone and retrieve the exact chapter
def query_vectors(query, selected_pdf):
    # Extract requested chapter/article number
    match = re.search(r'(CHAPTER|ARTICLE)\s+(\d+)', query, re.IGNORECASE)
    requested_section = f"{match.group(1).upper()} {match.group(2)}" if match else None
    
    print(f"🔍 Querying for: {requested_section}")  # Debugging

    # Query Pinecone
    results = index.query(
        vector=embedder.encode(query).tolist(),
        top_k=5,
        include_metadata=True,
        filter={"pdf_name": selected_pdf}
    )

    print("🔹 Pinecone Results:", results)  # Debugging

    if not results["matches"]:
        return "⚠️ No relevant information found in the selected document."

    # Retrieve only the requested chapter
    for match in results["matches"]:
        if requested_section and requested_section in match["metadata"].get("chapter", ""):
            return f"**Extracted Answer from {requested_section}:**\n\n{match['metadata']['text']}"

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
