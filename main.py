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

st.title("📜 Legal HelpDesk for Saudi Arabia")

# Select PDF Source
pdf_source = st.radio("📂 Select PDF Source", ["Upload from PC", "Choose from Document Storage"])

# Upload PDF
if pdf_source == "Upload from PC":
    uploaded_file = st.file_uploader("📤 Upload PDF File", type=["pdf"])
    
    if uploaded_file is not None:
        pdf_reader = PdfReader(uploaded_file)
        pdf_text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_text(pdf_text)
        
        # Store chunks in Pinecone
        vectors = [{"id": f"{uploaded_file.name}_{i}", "values": embedding_model.embed_query(chunk),
                    "metadata": {"pdf_name": uploaded_file.name, "text": chunk}} for i, chunk in enumerate(chunks)]
        
        index.upsert(vectors)
        st.success(f"✅ {uploaded_file.name} stored in Pinecone!")

# Select stored PDF from Pinecone
elif pdf_source == "Choose from Document Storage":
    # Retrieve stored PDFs
    stored_pdfs = set()
    results = index.query(vector=[0] * 384, top_k=100, include_metadata=True)
    for match in results["matches"]:
        stored_pdfs.add(match["metadata"]["pdf_name"])

    if stored_pdfs:
        selected_pdf = st.selectbox("📂 Choose a PDF Document", list(stored_pdfs))
    else:
        st.warning("⚠️ No PDFs found in storage.")
        selected_pdf = None

# Choose input and response language
input_lang = st.radio("🌍 Choose Input Language", ["English", "Arabic"], index=0)
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
                text = metadata.get("text", "No text available.")

                st.markdown(f"### 📌 Article from {pdf_name}")
                st.write(text)
        else:
            st.error("❌ No relevant legal articles found.")
