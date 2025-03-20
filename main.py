import os
import streamlit as st
import pinecone
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "helpdesk"

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Load embeddings model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Preloaded PDFs (Stored in Pinecone)
preloaded_pdfs = ["Law of the Council of Ministers.pdf"]

st.title("📜 Legal HelpDesk for Saudi Arabia")

# Dropdown to select PDF
selected_pdf = st.selectbox("📂 Choose a PDF Document", preloaded_pdfs)

# Choose input language
input_lang = st.radio("🌍 Choose Input Language", ["English", "Arabic"], index=0)

# Choose response language
response_lang = st.radio("🌍 Choose Response Language", ["English", "Arabic"], index=0)

# User query input
query = st.text_input("🔎 Ask a legal question:")

if st.button("Search in Legal Database"):
    if not query:
        st.warning("❗ Please enter a question.")
    else:
        # Embed query and search in Pinecone
        query_embedding = embedding_model.embed_query(query)
        results = index.query(vector=query_embedding, top_k=3, include_metadata=True)

        if results and results["matches"]:
            st.success("📖 Found relevant articles:")
            for match in results["matches"]:
                metadata = match["metadata"]
                pdf_name = metadata.get("pdf_name", "Unknown PDF")
                article_number = metadata.get("article_number", "Unknown")
                text = metadata.get("text", "No text available.")

                st.markdown(f"### 📌 Article {article_number} from {pdf_name}")
                st.write(text)
        else:
            st.error("❌ No relevant legal articles found.")
