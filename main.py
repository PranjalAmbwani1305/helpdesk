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
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index_name = "legal-helpdesk"

# Create index if not exists
if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, dimension=768, metric="cosine")
index = pinecone.Index(index_name)

# Load Sentence Transformer model for embeddings
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Function to extract structured chapters from PDF
def process_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

    # Split text into chapters using regex (assuming "Chapter X:" format)
    chapters = re.split(r'(CHAPTER \d+:)', text, flags=re.IGNORECASE)
    structured_data = {}

    for i in range(1, len(chapters), 2):
        chapter_title = chapters[i].strip().upper()
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
    match = re.search(r'\bCHAPTER (\d+)\b', query, re.IGNORECASE)
    
    if match:
        chapter_number = match.group(1)
        chapter_key = f"CHAPTER {chapter_number}:"
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
        return format_response(results["matches"][0]["metadata"]["title"], results["matches"][0]["metadata"]["text"])
    else:
        return "No relevant legal information found."

# Function to format response properly
def format_response(title, text):
    formatted_text = f"**{title}**\n\n{text}"
    return formatted_text

# Streamlit UI
st.markdown("<h1 style='text-align: center; color: #2E3B55;'>⚖️ AI-Powered Legal HelpDesk ⚖️</h1>", unsafe_allow_html=True)

pdf_source = st.radio("Select PDF Source", ["Upload from PC"])
selected_pdf = None

if pdf_source == "Upload from PC":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        temp_pdf_path = f"temp_{uploaded_file.name}"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.read())
        
        structured_data = process_pdf(temp_pdf_path)
        store_vectors(structured_data, uploaded_file.name)
        st.success("✅ PDF uploaded and processed successfully!")
        selected_pdf = uploaded_file.name

# Language selection
input_lang = st.radio("Choose Input Language", ["English", "Arabic"], index=0)
response_lang = st.radio("Choose Response Language", ["English", "Arabic"], index=0)

# User query input
query = st.text_input("Ask a legal question (e.g., 'CHAPTER 5'):" if input_lang == "English" else "اطرح سؤالاً قانونياً (مثال: 'الفصل 5'): ")

if st.button("Get Legal Information"):
    if selected_pdf and query:
        # Translate query to English for processing
        detected_lang = GoogleTranslator(source="auto", target="en").translate(query)
        response = query_vectors(detected_lang, selected_pdf)

        # Translate response if needed
        if response_lang == "Arabic":
            response = GoogleTranslator(source="en", target="ar").translate(response)
            st.markdown(f"<div dir='rtl' style='text-align: right; color: #444;'>{response}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='color: #444;'>{response}</div>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a legal question and upload a PDF.")
