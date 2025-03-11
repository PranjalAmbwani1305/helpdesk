import streamlit as st
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import os

# Initialize Pinecone
PINECONE_API_KEY = "your-api-key"  # Replace with actual API key
INDEX_NAME = "helpdesk"
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sidebar - List stored PDFs in Pinecone
def get_stored_pdfs():
    try:
        query_results = index.query(
            vector=[0] * 384,  # Dummy vector, use appropriate dimension
            top_k=100,
            include_metadata=True
        )
        return list(set(match["metadata"].get("pdf_name", "Unknown") for match in query_results["matches"]))
    except Exception as e:
        return ["Error fetching PDFs"]

st.sidebar.header("📂 Stored PDFs in Pinecone")
stored_pdfs = get_stored_pdfs()
st.sidebar.write("\n".join(stored_pdfs))

# File uploader
st.header("📜 AI-Powered Legal HelpDesk")
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with open(os.path.join("./", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Uploaded: {uploaded_file.name}")

# Query system
st.subheader("Ask a legal question:")
query = st.text_input("Enter your question")

if st.button("Search") and query:
    query_vector = model.encode(query).tolist()
    results = index.query(vector=query_vector, top_k=5, include_metadata=True)
    
    if results["matches"]:
        for match in results["matches"]:
            st.write(f"### {match['metadata']['title']}")
            st.write(f"📖 {match['metadata']['text']}")
    else:
        st.error("No relevant answer found.")
