import streamlit as st
import pinecone
import PyPDF2
import os
import re
from sentence_transformers import SentenceTransformer

# Load Pinecone API Key
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "helpdesk"

if not PINECONE_API_KEY:
    st.error("Pinecone API key is missing. Set it as an environment variable.")
    st.stop()

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(INDEX_NAME)

# Load Hugging Face Model
model = SentenceTransformer("all-MiniLM-L6-v2")

def clear_old_vectors():
    """Deletes all vectors from Pinecone before inserting new ones."""
    try:
        index.delete(delete_all=True)
        st.info("✅ Old vectors cleared from Pinecone.")
    except pinecone.openapi_support.exceptions.NotFoundException:
        st.warning("⚠️ Attempted to delete vectors, but the index is empty or not found.")
    except Exception as e:
        st.error(f"Error while deleting vectors: {e}")

def store_vectors(chapters, articles, pdf_name):
    """Stores extracted chapters and articles in Pinecone."""
    clear_old_vectors()  # Ensure only one PDF is stored at a time
    
    for i, chapter in enumerate(chapters):
        chapter_vector = model.encode(chapter['content']).tolist()
        index.upsert([
            (f"{pdf_name}-chapter-{i}", chapter_vector, 
            {"pdf_name": pdf_name, "text": chapter['content'], "type": "chapter"})
        ])
    
    for i, article in enumerate(articles):
        article_vector = model.encode(article['content']).tolist()
        index.upsert([
            (f"{pdf_name}-article-{i}", article_vector, 
            {"pdf_name": pdf_name, "chapter": article['chapter'], "text": article['content'], "type": "article"})
        ])
    
    st.success("✅ PDF content successfully stored in Pinecone.")
