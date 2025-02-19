import streamlit as st
import pinecone
import PyPDF2
import os
from dotenv import load_dotenv
from deep_translator import GoogleTranslator 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index_name = "helpdesk"

if index_name not in [i['name'] for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=768,  # Corrected dimension for BAAI/bge-base-en
        metric="cosine"
    )

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
index = Pinecone(index_name=index_name, embedding_function=embedding_model)

pdf_storage = {}

def process_pdf(pdf_path, chunk_size=500):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def store_vectors(chunks, pdf_name):
    for i, chunk in enumerate(chunks):
        try:
            index.add_texts([str(chunk)], metadatas=[{"pdf_name": pdf_name}])  # Removed "text"
        except Exception as e:
            st.error(f"Error adding vectors to Pinecone: {e}")

def query_vectors(query, selected_pdf):
    try:
        results = index.similarity_search(query, k=5, filter={"pdf_name": {"$eq": selected_pdf}})  # Fixed filter
        return "\n\n".join([res.page_content for res in results]) if results else "No relevant information found."
    except Exception as e:
        st.error(f"Error querying Pinecone: {e}")
        return "An error occurred."

def translate_text(text, target_language):
    return GoogleTranslator(source="auto", target=target_language).translate(text)

st.markdown("<h1 style='text-align: center;'>AI-Powered Legal HelpDesk</h1>", unsafe_allow_html=True)

st.sidebar.header("📂 Stored PDFs")
pdf_list = list(pdf_storage.keys())
if pdf_list:
    with st.sidebar.expander("📜 View Stored PDFs", expanded=False):
        for pdf in pdf_list:
            st.sidebar.write(f"📄 {pdf}")
else:
    st.sidebar.write("No PDFs stored yet. Upload one!")

selected_pdf = None

pdf_source = st.radio("Select PDF Source", ["Upload from PC", "Choose from Storage"])

if pdf_source == "Upload from PC":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        temp_pdf_path = f"temp_{uploaded_file.name}"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.read())
        
        chunks = process_pdf(temp_pdf_path)
        pdf_storage[uploaded_file.
