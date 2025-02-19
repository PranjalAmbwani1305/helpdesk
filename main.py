import streamlit as st
import os
import pdfplumber
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator
from sklearn.metrics.pairwise import cosine_similarity

# Load Hugging Face model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Directory for storing PDFs
PDF_STORAGE_DIR = "pdf_repository"
if not os.path.exists(PDF_STORAGE_DIR):
    os.makedirs(PDF_STORAGE_DIR)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n" if page.extract_text() else ""
    return text

# Function to split text into chunks
def chunk_text(text, chunk_size=200):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Function to store PDF embeddings
def store_pdf_embeddings(pdf_name, pdf_text):
    text_chunks = chunk_text(pdf_text)
    embeddings = model.encode(text_chunks, convert_to_tensor=False)

    storage_path = os.path.join(PDF_STORAGE_DIR, f"{pdf_name}.pkl")
    with open(storage_path, "wb") as f:
        pickle.dump({"chunks": text_chunks, "embeddings": embeddings}, f)

# Function to check if a PDF is already stored
def is_pdf_stored(pdf_name):
    return os.path.exists(os.path.join(PDF_STORAGE_DIR, f"{pdf_name}.pkl"))

# Function to load stored PDF embeddings
def load_pdf_embeddings(pdf_name):
    storage_path = os.path.join(PDF_STORAGE_DIR, f"{pdf_name}.pkl")
    with open(storage_path, "rb") as f:
        return pickle.load(f)

# UI Title
st.markdown("<h1 style='text-align: center;'>AI-Powered Legal HelpDesk</h1>", unsafe_allow_html=True)

# Sidebar for stored PDFs
st.sidebar.header("📂 Stored PDFs")
stored_pdfs = [f.replace(".pkl", "") for f in os.listdir(PDF_STORAGE_DIR) if f.endswith(".pkl")]
selected_pdf = st.sidebar.selectbox("Choose a stored PDF", stored_pdfs) if stored_pdfs else None

# File Upload Section
st.header("📂 Upload a PDF")
uploaded_file = st.file_uploader("Upload a legal document (PDF)", type=["pdf"])

if uploaded_file:
    pdf_name = uploaded_file.name
    pdf_path = os.path.join(PDF_STORAGE_DIR, pdf_name)

    if not is_pdf_stored(pdf_name):
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        pdf_text = extract_text_from_pdf(pdf_path)
        store_pdf_embeddings(pdf_name, pdf_text)
        st.success("PDF uploaded, processed, and stored!")
    else:
        st.info("PDF is already stored.")

# Choose input & response language
input_lang = st.radio("Choose Input Language", ["English", "Arabic"], index=0)
response_lang = st.radio("Choose Response Language", ["English", "Arabic"], index=0)

# User Query
query = st.text_input("Ask a question (in English or Arabic):" if input_lang == "English" else "اسأل سؤالاً (باللغة العربية أو الإنجليزية):")

if st.button("Get Answer"):
    if selected_pdf and query:
        # Load stored PDF text and embeddings
        pdf_data = load_pdf_embeddings(selected_pdf)
        text_chunks = pdf_data["chunks"]
        pdf_embeddings = pdf_data["embeddings"]

        # Get query embedding
        query_embedding = model.encode(query, convert_to_tensor=False)

        # Find top 3 most relevant passages
        similarity_scores = cosine_similarity([query_embedding], pdf_embeddings)[0]
        top_indices = np.argsort(similarity_scores)[-3:][::-1]  # Get top 3 matches

        # Retrieve top matching chunks
        matched_texts = "\n\n".join([text_chunks[i] for i in top_indices])

        # Translate response if needed
        if response_lang == "Arabic":
            matched_texts = GoogleTranslator(source="auto", target="ar").translate(matched_texts)
            st.markdown(f"<div dir='rtl' style='text-align: right;'>{matched_texts}</div>", unsafe_allow_html=True)
        else:
            st.write(f"**Answer:** {matched_texts}")
    else:
        st.warning("Please enter a query and select a stored PDF.")
