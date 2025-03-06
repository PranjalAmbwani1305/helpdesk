import streamlit as st
from pinecone import Pinecone
import os
import fitz  # PyMuPDF
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer

# Initialize Pinecone
PINECONE_API_KEY = "YOUR_PINECONE_API_KEY"
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "helpdesk"
index = pc.Index(index_name)

# Load Hugging Face Model (Change model here if needed)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_path):
    """Extracts text from the PDF file."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text

def query_vectors(query, selected_pdf):
    """Queries Pinecone for the most relevant result."""
    query_embedding = model.encode(query).tolist()
    response = index.query(vector=query_embedding, top_k=3, include_metadata=True)
    if response and 'matches' in response:
        return response['matches'][0]['metadata'].get('text', "No relevant answer found.")
    return "No relevant answer found."

def translate_text(text, target_lang):
    """Translates text using GoogleTranslator."""
    return GoogleTranslator(source="auto", target=target_lang).translate(text)

# Streamlit UI
st.title("AI-Powered Legal HelpDesk for Saudi Arabia")

# Sidebar for Stored PDFs
st.sidebar.header("📂 Stored PDFs")
stored_pdfs = ["Basic Law Governance.pdf", "Law of the Consultative Council.pdf", "Law of the Council of Ministers.pdf"]
for pdf in stored_pdfs:
    st.sidebar.markdown(f"📄 {pdf}")

# PDF Source Selection
pdf_source = st.radio("Select PDF Source", ["Upload from PC", "Choose from the Document Storage"], index=1)
selected_pdf = None

if pdf_source == "Upload from PC":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file is not None:
        temp_pdf_path = os.path.join("/tmp", uploaded_file.name)
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.read())
        selected_pdf = uploaded_file.name
else:
    selected_pdf = st.selectbox("Select a PDF", stored_pdfs)

# Language Selection
input_lang = st.radio("Choose Input Language", ["English", "Arabic"], index=0)
response_lang = st.radio("Choose Response Language", ["English", "Arabic"], index=0)

# Query Input
query = st.text_input("Ask a question (in English or Arabic):")

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
