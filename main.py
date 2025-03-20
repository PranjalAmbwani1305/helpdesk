import os
import streamlit as st
import pinecone
from sentence_transformers import SentenceTransformer

# Load environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "helpdesk"

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Load SentenceTransformer model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Streamlit UI
st.title("Legal HelpDesk: Article Search")

# User query input
query = st.text_input("Enter the article number or keyword:")

# Process User Query
if query:
    try:
        # Generate query vector
        query_vector = embedding_model.encode(query).tolist()

        # Search in Pinecone
        results = index.query(vector=query_vector, top_k=1, include_metadata=True)  # Get only top 1 match

        # Display only the article text
        if results["matches"]:
            best_match = results["matches"][0]  # Get the top-ranked match

            # Extract only the article text
            article_text = best_match["metadata"].get("text", "No content available.")
            
            # ✅ Display only the article text
            st.write("### Article Text")
            st.write(article_text)

        else:
            st.info("No relevant article found.")

    except Exception as e:
        st.error(f"Error retrieving results: {e}")
