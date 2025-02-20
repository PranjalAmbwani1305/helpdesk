import streamlit as st
import pdfplumber
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from bson import ObjectId

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client["helpdesk"]
collection = db["data"]

# Load Hugging Face embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def process_pdf(pdf_path, chunk_size=500):
    """Extracts and chunks text from a PDF."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += (page.extract_text() or "") + "\n"

    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def store_vectors(chunks, pdf_name):
    """Embeds and stores chunks in MongoDB."""
    vectors = embedding_model.encode(chunks).tolist()

    documents = [
        {"pdf_name": pdf_name, "chunk_id": i, "embedding": vectors[i], "text": chunks[i]}
        for i in range(len(chunks))
    ]
    
    collection.insert_many(documents)

def query_vectors(query, selected_pdf):
    """Searches for relevant text using MongoDB."""
    query_vector = embedding_model.encode([query]).tolist()[0]
    
    # Retrieve all chunks for the selected PDF
    docs = list(collection.find({"pdf_name": selected_pdf}, {"embedding": 1, "text": 1}))

    # Compute similarity scores (cosine similarity)
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    stored_vectors = np.array([doc["embedding"] for doc in docs])
    query_vector = np.array(query_vector).reshape(1, -1)
    
    similarities = cosine_similarity(query_vector, stored_vectors)[0]
    top_indices = similarities.argsort()[-5:][::-1]  # Get top 5 matches

    matched_texts = [docs[i]["text"] for i in top_indices]
    
    return "\n\n".join(matched_texts) if matched_texts else "No relevant information found."

# Streamlit UI
st.title("🔍 AI-Powered Legal HelpDesk")

pdf_source = st.radio("Select PDF Source", ["Upload from PC"])
selected_pdf = None

if pdf_source == "Upload from PC":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        temp_pdf_path = f"temp_{uploaded_file.name}"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.read())
        
        chunks = process_pdf(temp_pdf_path)
        store_vectors(chunks, uploaded_file.name)
        st.success("PDF uploaded and processed!")
        selected_pdf = uploaded_file.name

        os.remove(temp_pdf_path)  # Clean up temp file

query = st.text_input("Ask a legal question:")
if st.button("Get Answer"):
    if selected_pdf and query:
        with st.spinner("Searching..."):
            response = query_vectors(query, selected_pdf)
        st.write(f"**Answer:** {response}")
    else:
        st.warning("Please upload a PDF and enter a question.")
