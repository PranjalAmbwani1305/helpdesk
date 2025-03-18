import os
import pinecone
import streamlit as st
import tempfile

# Load Pinecone API Key
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "helpdesk"

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment="us-west1-gcp")  # Adjust environment as needed
index = pinecone.Index(INDEX_NAME)

# Function to fetch stored PDFs
def get_stored_pdfs():
    try:
        response = index.describe_index_stats()
        vector_count = response["total_vector_count"]

        stored_pdfs = []
        if vector_count > 0:
            vectors = index.query(vector_ids=[], top_k=vector_count, include_metadata=True)
            for vector in vectors["matches"]:
                if "metadata" in vector and "filename" in vector["metadata"]:
                    stored_pdfs.append(vector["metadata"]["filename"])
        return stored_pdfs
    except Exception as e:
        st.error(f"Error fetching stored PDFs: {e}")
        return []

# Streamlit UI
st.title("⚖️ AI-Powered Legal HelpDesk for Saudi Arabia")

# Sidebar for stored PDFs
