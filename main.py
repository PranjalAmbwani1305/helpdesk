import streamlit as st
import pinecone
import PyPDF2
import os
import re
import time
import asyncio
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer

# AsyncIO Loop Fix for Python 3.12
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Read API Key from Streamlit Secrets
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets.get("PINECONE_ENV", "us-east-1")  # Default Region

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index_name = "helpdesk"

# Check if Index Exists
if index_name not in pinecone.list_indexes():
    print("⚠️ Index does not exist. Creating index...")
    pinecone.create_index(name=index_name, dimension=384, metric="cosine")
    time.sleep(10)  # Wait for index to be ready

index = pinecone.Index(index_name)
print("✅ Pinecone Index Ready:", index.describe_index_stats())

# Load Sentence Transformer Model
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Function to Extract Text from PDF
def process_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

    # Split Text into Chapters using Regex
    chapters = re.split(r'(CHAPTER\s+\d+|ARTICLE\s+\d+)', text, flags=re.IGNORECASE)
    structured_data = {}

    for i in range(1, len(chapters), 2):
        title = chapters[i].strip()
        content = chapters[i + 1].strip() if i + 1 < len(chapters) else ""
        structured_data[title] = content

    return structured_data

# Function to Check Existing PDFs in Pinecone
def get_existing_pdfs():
    existing_pdfs = set()
    try:
        stats = index.describe_index_stats()
        if stats["total_vector_count"] == 0:
            return existing_pdfs
        
        results = index.query(vector=embedder.encode("sample").tolist(), top_k=10, include_metadata=True)
        for match in results["matches"]:
            existing_pdfs.add(match["metadata"]["pdf_name"])
    except Exception as e:
        print("⚠️ Error Checking Pinecone Data:", e)

    return existing_pdfs

# Store Embeddings in Pinecone
def store_vectors(structured_data, pdf_name):
    existing_pdfs = get_existing_pdfs()
    if pdf_name in existing_pdfs:
        print(f"⚠️ {pdf_name} already exists. Skipping storage.")
        return

    for title, content in structured_data.items():
        vector = embedder.encode(content).tolist()
        metadata = {"pdf_name": pdf_name, "chapter": title, "text": content}

        index.upsert([(f"{pdf_name}-{title}", vector, metadata)])
        print(f"✅ Stored: {title} in Pinecone")

# Function to Query Pinecone
def query_vectors(query, selected_pdf):
    vector = embedder.encode(query).tolist()

    results = index.query(vector=vector, top_k=5, include_metadata=True, filter={"pdf_name": {"$eq": selected_pdf}})
    
    if not results["matches"]:
        return "⚠️ No relevant information found."

    for match in results["matches"]:
        chapter = match["metadata"]["chapter"]
        text = match["metadata"]["text"]
        return f"**Extracted Answer from {chapter}:**\n\n{text[:1000]}"

    return "⚠️ No matching chapter found."

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>📜 AI-Powered Legal HelpDesk</h1>", unsafe_allow_html=True)

action = st.radio("Choose an Action:", ["Upload PDF", "Query Existing PDFs"])

if action == "Upload PDF":
    uploaded_file = st.file_uploader("📂 Upload a PDF", type=["pdf"])
    if uploaded_file:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.read())

        structured_data = process_pdf(uploaded_file.name)
        store_vectors(structured_data, uploaded_file.name)
        st.success("✅ PDF Uploaded and Stored Successfully!")

# List Existing PDFs
existing_pdfs = get_existing_pdfs()
if existing_pdfs:
    selected_pdf = st.selectbox("📄 Select Existing PDF", list(existing_pdfs))
else:
    selected_pdf = None
    st.warning("⚠️ No PDFs Found in Pinecone. Please Upload a PDF.")

# Language Options
input_lang = st.radio("🌐 Input Language", ["English", "Arabic"])
response_lang = st.radio("🌐 Response Language", ["English", "Arabic"])

# Query Section
query = st.text_input("🔍 Ask Your Legal Question:")
if st.button("Get Answer"):
    if selected_pdf and query:
        translated_query = GoogleTranslator(source="auto", target="en").translate(query)
        response = query_vectors(translated_query, selected_pdf)

        if response_lang == "Arabic":
            response = GoogleTranslator(source="en", target="ar").translate(response)
            st.markdown(f"<div dir='rtl'>{response}</div>", unsafe_allow_html=True)
        else:
            st.markdown(response)
    else:
        st.warning("⚠️ Please Enter a Query and Select a PDF.")

# Debugging Pinecone Storage
if st.button("🛠️ Debug Pinecone Storage"):
    print("📌 Pinecone Index Stats:", index.describe_index_stats())
