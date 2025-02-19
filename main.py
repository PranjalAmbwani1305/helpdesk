import streamlit as st
import os
import requests
import pdfkit
import PyPDF2
from sentence_transformers import SentenceTransformer
import pickle
from deep_translator import GoogleTranslator
from bs4 import BeautifulSoup

# Load Hugging Face model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Directory for storing PDFs
PDF_STORAGE_DIR = "pdf_repository"
if not os.path.exists(PDF_STORAGE_DIR):
    os.makedirs(PDF_STORAGE_DIR)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    return text

# Function to store PDF text embeddings
def store_pdf_embeddings(pdf_name, pdf_text):
    embeddings = model.encode(pdf_text, convert_to_tensor=False)
    storage_path = os.path.join(PDF_STORAGE_DIR, f"{pdf_name}.pkl")
    with open(storage_path, "wb") as f:
        pickle.dump({"text": pdf_text, "embeddings": embeddings}, f)

# Function to check if a PDF is already stored
def is_pdf_stored(pdf_name):
    return os.path.exists(os.path.join(PDF_STORAGE_DIR, f"{pdf_name}.pkl"))

# Function to load stored PDF embeddings
def load_pdf_embeddings(pdf_name):
    storage_path = os.path.join(PDF_STORAGE_DIR, f"{pdf_name}.pkl")
    with open(storage_path, "rb") as f:
        return pickle.load(f)

# Function to scrape a webpage and save as PDF
def scrape_webpage_to_pdf(url, pdf_name):
    try:
        response = requests.get(url)
        if response.status_code != 200:
            return f"Error: Unable to fetch the webpage. Status Code: {response.status_code}"
        
        soup = BeautifulSoup(response.text, "html.parser")
        page_text = soup.get_text(separator="\n")

        # Save as PDF
        pdf_path = os.path.join(PDF_STORAGE_DIR, pdf_name)
        pdfkit.from_string(page_text, pdf_path)
        
        return pdf_path
    except Exception as e:
        return f"Error: {str(e)}"

# UI Title
st.markdown("<h1 style='text-align: center;'>AI-Powered Legal HelpDesk</h1>", unsafe_allow_html=True)

# Sidebar for stored PDFs
st.sidebar.header("📂 Stored PDFs")
stored_pdfs = [f.replace(".pkl", "") for f in os.listdir(PDF_STORAGE_DIR) if f.endswith(".pkl")]
if stored_pdfs:
    selected_pdf = st.sidebar.selectbox("Choose a stored PDF", stored_pdfs)
else:
    selected_pdf = None

# Web Scraper Input
st.header("🔗 Scrape a Webpage")
webpage_url = st.text_input("Enter the URL of the legal webpage:")
scrape_button = st.button("Scrape and Save as PDF")

if scrape_button and webpage_url:
    pdf_name = f"{webpage_url.replace('https://', '').replace('http://', '').replace('/', '_')}.pdf"
    
    if not is_pdf_stored(pdf_name):
        pdf_path = scrape_webpage_to_pdf(webpage_url, pdf_name)
        if "Error" in pdf_path:
            st.error(pdf_path)
        else:
            pdf_text = extract_text_from_pdf(pdf_path)
            store_pdf_embeddings(pdf_name, pdf_text)
            st.success(f"Webpage scraped and stored as {pdf_name}!")
    else:
        st.info("This webpage has already been scraped and stored.")

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
if input_lang == "Arabic":
    query = st.text_input("اسأل سؤالاً (باللغة العربية أو الإنجليزية):")
    st.markdown("<style>.stTextInput>div>div>input {direction: rtl; text-align: right;}</style>", unsafe_allow_html=True)
else:
    query = st.text_input("Ask a question (in English or Arabic):")

# Process query
if st.button("Get Answer"):
    if selected_pdf and query:
        # Load stored PDF text and embeddings
        pdf_data = load_pdf_embeddings(selected_pdf)
        pdf_text = pdf_data["text"]
        pdf_embeddings = pdf_data["embeddings"]

        # Get query embedding
        query_embedding = model.encode(query, convert_to_tensor=False)

        # Find the most relevant passage using cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_scores = cosine_similarity([query_embedding], [pdf_embeddings])[0]
        best_match_index = similarity_scores.argmax()

        # Retrieve best matching text snippet
        matched_text = pdf_text.split("\n")[best_match_index]

        # Translate response if needed
        if response_lang == "Arabic":
            matched_text = GoogleTranslator(source="auto", target="ar").translate(matched_text)
            st.markdown(f"<div dir='rtl' style='text-align: right;'>{matched_text}</div>", unsafe_allow_html=True)
        else:
            st.write(f"**Answer:** {matched_text}")
    else:
        st.warning("Please enter a query and select a stored PDF.")
