import streamlit as st
import pinecone
import PyPDF2
import os
from dotenv import load_dotenv
from deep_translator import GoogleTranslator  

load_dotenv()

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# Function to process PDF and extract text
def process_pdf(pdf_path, chunk_size=500):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

# Function to store PDF in local directory
def store_pdf(pdf_name, pdf_data):
    pdf_path = os.path.join("stored_pdfs", pdf_name)
    with open(pdf_path, "wb") as f:
        f.write(pdf_data)

# Function to list stored PDFs
def list_stored_pdfs():
    if not os.path.exists("stored_pdfs"):
        os.makedirs("stored_pdfs")
    return [f for f in os.listdir("stored_pdfs") if f.endswith(".pdf")]

# Function to translate text
def translate_text(text, target_language):
    return GoogleTranslator(source="auto", target=target_language).translate(text)

st.markdown("""
    <h1 style='text-align: center;'>AI-Powered Legal HelpDesk</h1>
""", unsafe_allow_html=True)

st.sidebar.header("📂 Stored PDFs")
pdf_list = list_stored_pdfs()
if pdf_list:
    with st.sidebar.expander("📜 View Stored PDFs", expanded=False):
        for pdf in pdf_list:
            st.sidebar.write(f"📄 {pdf}")
else:
    st.sidebar.write("No PDFs stored yet. Upload one!")

selected_pdf = None
pdf_source = st.radio("Select PDF Source", ["Upload from PC", "Choose from the Document Storage"])

if pdf_source == "Upload from PC":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        pdf_path = os.path.join("stored_pdfs", uploaded_file.name)
        store_pdf(uploaded_file.name, uploaded_file.read())
        st.success("PDF uploaded and stored locally!")
        selected_pdf = uploaded_file.name

elif pdf_source == "Choose from the Document Storage":
    if pdf_list:
        selected_pdf = st.selectbox("Select a PDF", pdf_list)
    else:
        st.warning("No PDFs available in storage. Please upload one.")

input_lang = st.radio("Choose Input Language", ["English", "Arabic"], index=0)
response_lang = st.radio("Choose Response Language", ["English", "Arabic"], index=0)

if input_lang == "Arabic":
    query = st.text_input("اسأل سؤالاً (باللغة العربية أو الإنجليزية):", key="query_input")
    query_html = """
    <style>
    .stTextInput>div>div>input {
        direction: rtl;
        text-align: right;
    }
    </style>
    """
    st.markdown(query_html, unsafe_allow_html=True)
else:
    query = st.text_input("Ask a question (in English or Arabic):", key="query_input")

if st.button("Get Answer"):
    if selected_pdf and query:
        pdf_path = os.path.join("stored_pdfs", selected_pdf)
        pdf_chunks = process_pdf(pdf_path)
        response = "\n".join(pdf_chunks[:3])  # Mock response with first few chunks
        
        if response_lang == "Arabic":
            response = translate_text(response, "ar")
            st.markdown(f"<div dir='rtl' style='text-align: right;'>{response}</div>", unsafe_allow_html=True)
        else:
            st.write(f"**Answer:** {response}")
    else:
        st.warning("Please enter a query and select a PDF.")

