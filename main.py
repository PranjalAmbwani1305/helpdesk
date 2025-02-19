import streamlit as st
import pinecone
import torch
import PyPDF2
import os
from dotenv import load_dotenv
from deep_translator import GoogleTranslator  
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()


HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
embed_model = SentenceTransformer(HUGGINGFACE_MODEL)


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# Connect to existing index
index = pc.Index(PINECONE_INDEX)

# Function to process PDFs into chunks
def process_pdf(pdf_path, chunk_size=500):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

# Function to store PDFs locally
def store_pdf(pdf_name, pdf_data):
    pdf_dir = "stored_pdfs"
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, pdf_name)
    with open(pdf_path, "wb") as f:
        f.write(pdf_data)

# Function to list stored PDFs
def list_stored_pdfs():
    pdf_dir = "stored_pdfs"
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
    return [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

# Function to query Pinecone and retrieve relevant text chunks
def query_vectors(query, selected_pdf):
    query_embedding = embed_model.encode(query).tolist()
    
    results = index.query(vector=query_embedding, top_k=5, include_metadata=True, filter={"pdf_name": {"$eq": selected_pdf}})
    
    if results["matches"]:
        matched_texts = [match["metadata"]["text"] for match in results["matches"]]
        return "\n\n".join(matched_texts)
    else:
        return "No relevant information found in the selected document."

# Function to translate text
def translate_text(text, target_language):
    return GoogleTranslator(source="auto", target=target_language).translate(text)

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>AI-Powered Legal HelpDesk</h1>", unsafe_allow_html=True)

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
        detected_lang = GoogleTranslator(source="auto", target="en").translate(query)
        response = query_vectors(detected_lang, selected_pdf)

        if response_lang == "Arabic":
            response = translate_text(response, "ar")
            st.markdown(f"<div dir='rtl' style='text-align: right;'>{response}</div>", unsafe_allow_html=True)
        else:
            st.write(f"**Answer:** {response}")
    else:
        st.warning("Please enter a query and select a PDF.")
