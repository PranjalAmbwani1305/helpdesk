import streamlit as st
import pinecone
import os
from sentence_transformers import SentenceTransformer

# ✅ Pinecone Initialization (Using Environment Variable)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "helpdesk"

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Load SentenceTransformer model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Streamlit UI
st.title("📜 AI-Powered Legal Helpdesk")

# PDF Source Selection
pdf_source = st.radio("Select PDF Source", ["Upload from PC", "Choose from the Document Storage"])

# Handle PDF Selection from Storage
if pdf_source == "Choose from the Document Storage":
    st.subheader("📚 Select a PDF")
    
    # Fetch stored PDFs from Pinecone (Modify as needed to fetch actual data)
    stored_pdfs = ["Basic Law Governance.pdf", "www.saudiembassy.netlaw-provinces.pdf"]  # Example list
    selected_pdf = st.selectbox("Select a PDF", stored_pdfs)

# Choose Input & Response Language
st.subheader("🌍 Choose Input & Response Language")
input_language = st.radio("Choose Input Language", ["English", "Arabic"])
response_language = st.radio("Choose Response Language", ["English", "Arabic"])

# User Input for Legal Questions
st.subheader("🔍 Ask a Question")
query = st.text_input("Enter your legal question:")

# Process User Query
if query:
    try:
        # Generate query vector
        query_vector = embedding_model.encode(query).tolist()

        # Search in Pinecone
        results = index.query(vector=query_vector, top_k=5, include_metadata=True)

        # Display relevant legal articles
        if results["matches"]:
            st.subheader("📖 Relevant Legal Articles:")

            for match in results["matches"]:
                pdf_name = match["metadata"].get("pdf_name", "Unknown Document")
                article_number = match["metadata"].get("article_number", "Unknown")
                text = match["metadata"].get("text", "No content available.")

                # ✅ **Structured Answer Format**
                response_text = f"""
                **Article {article_number} of the provided legal document pertains to:**  
                {text}  

                🔹 **Explanation:**  
                - This article outlines the leadership structure of the government.  
                - It ensures that all resolutions require the King’s approval before being finalized.
                """
                st.write(response_text)
        else:
            st.info("No relevant articles found.")

    except Exception as e:
        st.error(f"Error retrieving results: {e}")
