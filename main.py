import streamlit as st
import pinecone
import PyPDF2
import os
import re
import time
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

# Ensure Index Exists Before Querying
if index_name not in pc.list_indexes().names():
    print("⚠️ Index does not exist. Creating index...")
    pc.create_index(name=index_name, dimension=768, metric="cosine")

# Wait for index to be ready before querying
time.sleep(5)  # Wait 5 seconds for the index to be ready

index = pc.Index(index_name)
print("✅ Pinecone Index Ready:", index.describe_index_stats())

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
        
        # Ensure correct metadata storage
        metadata = {"pdf_name": pdf_name, "chapter": title, "text": content}
        
        print(f"📌 Storing: {title} in Pinecone with {len(vector)} dimensions")
        index.upsert([(f"{pdf_name}-{title}", vector, metadata)])

# Function to check if Pinecone is storing data properly
def debug_pinecone_storage():
    print("📌 Checking Pinecone index stats:")
    print(index.describe_index_stats())

    try:
        stored_data = index.query(vector=[0]*768, top_k=5, include_metadata=True)
        print("📌 Sample stored data:", stored_data)
    except Exception as e:
        print("⚠️ Pinecone Storage Check Failed:", str(e))

# Function to query Pinecone and retrieve the exact chapter
def query_vectors(query, selected_pdf):
    vector = embedder.encode(query).tolist()

    if len(vector) != 768:  # Adjust if using a different model
        raise ValueError(f"⚠️ Query vector has incorrect dimensions! Expected 768, got {len(vector)}")

    print(f"🔍 Querying Pinecone for: {query}")
    print(f"📌 Using vector of length: {len(vector)}")

    try:
        results = index.query(
            vector=vector, 
            top_k=5, 
            include_metadata=True, 
            filter={"pdf_name": selected_pdf}
        )

        print("📌 Pinecone Query Results:", results)  # Debugging

        if not results["matches"]:
            return "⚠️ No relevant information found in the selected document."

        matched_texts = [match["metadata"]["text"] for match in results["matches"]]
        return "\n\n".join(matched_texts)

    except Exception as e:
        print("⚠️ Pinecone Query Failed:", str(e))
        return "⚠️ Error occurred while querying Pinecone."

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
