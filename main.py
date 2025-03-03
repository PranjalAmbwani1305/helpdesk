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

index = pc.Index(index_name)
# Load Sentence Transformer model for embeddings
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Function to extract structured chapters from PDF
def process_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

    # Split text into chapters using regex (assuming "Chapter X:" format)
    chapters = re.split(r'(Chapter \d+:)', text)
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
        index.upsert([(f"{pdf_name}-{title}", vector, {"pdf_name": pdf_name, "title": title, "text": content})])

# Function to query Pinecone and retrieve the exact chapter
def query_vectors(query, selected_pdf):
    # Check if user is asking for a specific chapter
    match = re.search(r'\bChapter (\d+)\b', query, re.IGNORECASE)
    
    if match:
        chapter_number = match.group(1)
        chapter_key = f"Chapter {chapter_number}:"
        results = index.query(
            vector=embedder.encode(chapter_key).tolist(),
            top_k=1,
            include_metadata=True,
            filter={"pdf_name": {"$eq": selected_pdf}, "title": {"$eq": chapter_key}}
        )
    else:
        # Default semantic search for general questions
        results = index.query(
            vector=embedder.encode(query).tolist(),
            top_k=5,
            include_metadata=True,
            filter={"pdf_name": {"$eq": selected_pdf}}
        )

    if results["matches"]:
        return results["matches"][0]["metadata"]["text"]  # Return the exact chapter content
    else:
        return "No relevant information found."

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
