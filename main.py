import os
import pinecone
import streamlit as st
from sentence_transformers import SentenceTransformer

# Load Pinecone API Key and Index Name
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "helpdesk"

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Load SentenceTransformer model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Streamlit UI
st.title("📖 Legal HelpDesk for Saudi Arabia")

# Section: Select PDF Source
st.header("Select PDF Source")
pdf_source = st.radio("Choose:", ["Upload from PC", "Choose from the Document Storage"])

# PDF Selection (from stored PDFs in Pinecone metadata)
if pdf_source == "Choose from the Document Storage":
    # Retrieve stored PDFs
    existing_docs = index.describe_index_stats().get("namespaces", {})
    available_pdfs = list(existing_docs.keys())

    if available_pdfs:
        selected_pdf = st.selectbox("Select a PDF", available_pdfs)
    else:
        st.warning("⚠️ No PDFs found in the database.")
        selected_pdf = None

# Language Selection
st.header("Choose Input Language")
input_language = st.radio("", ["English", "Arabic"])

st.header("Choose Response Language")
response_language = st.radio("", ["English", "Arabic"])

# User Query Input
st.header("🔍 Ask a question (in English or Arabic)")
query = st.text_input("Enter your legal question:")

# Search Button
if st.button("Search in Legal Database"):
    if not query:
        st.warning("❗ Please enter a question.")
    elif not selected_pdf:
        st.warning("⚠️ Please select a PDF from storage.")
    else:
        # Embed query and search in Pinecone with filter
        query_embedding = embedding_model.encode(query).tolist()
        results = index.query(
            vector=query_embedding,
            top_k=3,
            include_metadata=True,
            filter={"pdf_name": selected_pdf}  # Ensure results come only from the selected PDF
        )

        if results and results["matches"]:
            st.success(f"📖 Found relevant articles from {selected_pdf}:")
            for match in results["matches"]:
                metadata = match["metadata"]
                text = metadata.get("text", "No text available.")
                st.markdown(f"### 📌 Article from {selected_pdf}")
                st.write(text)
        else:
            st.error("❌ No relevant legal articles found in this document.")
