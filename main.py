import streamlit as st
import os
import pinecone
from PyPDF2 import PdfReader
import hashlib
from deep_translator import GoogleTranslator

# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "helpdesk"

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Function to generate a unique ID for each document
def generate_pdf_id(pdf_name):
    return hashlib.md5(pdf_name.encode()).hexdigest()

# Function to upload PDF content to Pinecone
def upload_pdf_to_pinecone(file_name, file_content):
    pdf_id = generate_pdf_id(file_name)
    index.upsert(vectors=[
        {
            "id": pdf_id,
            "values": [0] * 384,  # Dummy vector
            "metadata": {"file_name": file_name, "content": file_content}
        }
    ])
    return pdf_id

# Function to fetch stored PDFs from Pinecone
def get_stored_pdfs():
    try:
        response = index.query(queries=[[0] * 384], top_k=50, include_metadata=True)
        return [match["metadata"]["file_name"] for match in response["matches"]]
    except Exception as e:
        st.error("Error fetching stored PDFs: " + str(e))
        return []

# Function for translation
def translate_text(text, src_lang, target_lang):
    if src_lang != target_lang:
        return GoogleTranslator(source=src_lang, target=target_lang).translate(text)
    return text

# UI Layout
st.title("📜 AI-Powered Legal HelpDesk for Saudi Arabia")

# Sidebar: Show previously uploaded PDFs
st.sidebar.header("📁 Uploaded PDFs")
stored_pdfs = get_stored_pdfs()
selected_pdf = st.sidebar.selectbox("Select a PDF:", stored_pdfs if stored_pdfs else ["No PDFs uploaded yet"])

# Language Selection
st.sidebar.header("🌍 Language Settings")
input_language = st.sidebar.radio("Choose Input Language:", ["English", "Arabic"])
response_language = st.sidebar.radio("Choose Response Language:", ["English", "Arabic"])

# File Upload Section
st.subheader("📂 Upload a PDF")
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    pdf_reader = PdfReader(uploaded_file)
    text_content = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    
    # Translate text if needed
    translated_text = translate_text(text_content, "en" if input_language == "English" else "ar", "en")

    # Store in Pinecone
    upload_pdf_to_pinecone(uploaded_file.name, translated_text)
    st.success(f"✅ {uploaded_file.name} uploaded successfully!")

# Ask a Question
st.subheader("❓ Ask a Question")
question = st.text_area("Enter your question:", "")

if st.button("Submit"):
    if question:
        translated_question = translate_text(question, "en" if input_language == "English" else "ar", "en")
        st.write(f"📝 Answer in {response_language}: {translated_question}")  # Placeholder for response
    else:
        st.warning("⚠️ Please enter a question!")

