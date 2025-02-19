import streamlit as st
import os
import PyPDF2
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator  

# Load environment variables
load_dotenv()

# Hugging Face Model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(EMBEDDING_MODEL)

# Storage dictionary for PDFs and text (replace with database if needed)
pdf_storage = {}

# Function to process PDF and extract text
def process_pdf(uploaded_file):
    text = ""
    reader = PyPDF2.PdfReader(uploaded_file)
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

# Function to get embeddings
def get_embedding(text):
    return model.encode(text).tolist()

# Function to translate text
def translate_text(text, target_language):
    return GoogleTranslator(source="auto", target=target_language).translate(text)

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>AI-Powered Legal HelpDesk</h1>", unsafe_allow_html=True)

# Sidebar for PDF uploads
st.sidebar.header("📂 Stored PDFs")
if pdf_storage:
    with st.sidebar.expander("📜 View Stored PDFs", expanded=False):
        for pdf in pdf_storage.keys():
            st.sidebar.write(f"📄 {pdf}")
else:
    st.sidebar.write("No PDFs stored yet. Upload one!")

# PDF Upload
uploaded_file = st.file_uploader("Upload a legal document (PDF)", type=["pdf"])

if uploaded_file:
    pdf_name = uploaded_file.name
    if pdf_name not in pdf_storage:
        pdf_text = process_pdf(uploaded_file)
        pdf_storage[pdf_name] = pdf_text
        st.success(f"PDF '{pdf_name}' uploaded and stored successfully!")
    else:
        st.info(f"PDF '{pdf_name}' already exists in storage.")

# Select PDF for querying
if pdf_storage:
    selected_pdf = st.selectbox("Select a stored PDF for querying", list(pdf_storage.keys()))
else:
    selected_pdf = None

# Input and Output Language Selection
input_lang = st.radio("Choose Input Language", ["English", "Arabic"], index=0)
response_lang = st.radio("Choose Response Language", ["English", "Arabic"], index=0)

# Query Input
if input_lang == "Arabic":
    query = st.text_input("اسأل سؤالاً (باللغة العربية أو الإنجليزية):", key="query_input")
    st.markdown("<style>.stTextInput>div>div>input { direction: rtl; text-align: right; }</style>", unsafe_allow_html=True)
else:
    query = st.text_input("Ask a question (in English or Arabic):", key="query_input")

# Answer Generation
if st.button("Get Answer") and selected_pdf and query:
    document_text = pdf_storage[selected_pdf]

    # Generate query embedding
    query_embedding = get_embedding(query)

    # Simple text retrieval (replace with actual NLP-based retrieval)
    sentences = document_text.split("\n")
    relevant_text = "\n".join(sentences[:5])  # Taking first 5 lines as an example

    response = f"Based on '{selected_pdf}', here is the relevant legal information:\n\n{relevant_text}"

    # Translate response if needed
    if response_lang == "Arabic":
        response = translate_text(response, "ar")
        st.markdown(f"<div dir='rtl' style='text-align: right;'>{response}</div>", unsafe_allow_html=True)
    else:
        st.write(f"**Answer:** {response}")
