import streamlit as st
from pinecone import Pinecone
import uuid
from sentence_transformers import SentenceTransformer
import os

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("your-index-name")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text, chunk_size=100):
    """Splits text into chunks of a given size."""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def store_in_pinecone(text_chunks):
    """Stores text chunks in Pinecone."""
    vectors = [(str(uuid.uuid4()), model.encode(chunk).tolist(), {"text": chunk}) for chunk in text_chunks]
    index.upsert(vectors)

# Streamlit UI
st.title("Pinecone Chunk Storage")
user_input = st.text_area("Enter your text:")
chunk_size = st.slider("Chunk Size", 50, 500, 100)

if st.button("Store in Pinecone"):
    if user_input:
        chunks = chunk_text(user_input, chunk_size)
        store_in_pinecone(chunks)
        st.success(f"Stored {len(chunks)} chunks in Pinecone!")
    else:
        st.error("Please enter some text.")
