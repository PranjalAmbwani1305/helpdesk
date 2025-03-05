import streamlit as st
import pinecone
import torch
from transformers import AutoModel, AutoTokenizer
from PyPDF2 import PdfReader

# ✅ Load Secrets for Pinecone API Key and Environment
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENV"]

# ✅ Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index_name = "helpdesk"
index = pc.Index(index_name)

# ✅ Load Transformer Model for Embedding
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# ✅ Function to Create Embeddings
def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output = model(**inputs)
    return output.last_hidden_state[:, 0, :].squeeze().tolist()

# ✅ Function to Chunk PDF by Chapter & Article Name
def chunk_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    chunks = []
    current_title = None
    current_content = []

    for page in reader.pages:
        text = page.extract_text()
        if not text:
            continue
        
        lines = text.split("\n")
        for line in lines:
            if line.strip().lower().startswith(("chapter", "article")):
                if current_title:  # ✅ Store previous chunk
                    chunks.append({"title": current_title, "content": "\n".join(current_content)})
                current_title = line.strip()
                current_content = []
            else:
                current_content.append(line.strip())

    if current_title:
        chunks.append({"title": current_title, "content": "\n".join(current_content)})

    return chunks

# ✅ Store Vectors in Pinecone
def store_vectors(chunks, pdf_name):
    for i, chunk in enumerate(chunks):
        vector = embed_text(chunk["content"])

        if not vector or not isinstance(vector, list):
            print(f"Skipping chunk '{chunk['title']}' due to missing vector!")
            continue

        metadata = {
            "pdf_name": str(pdf_name),
            "title": str(chunk["title"]),
            "text": str(chunk["content"])
        }

        try:
            index.upsert([(f"{pdf_name}-doc-{i}", vector, metadata)])
        except Exception as e:
            print(f"Error inserting into Pinecone: {e}")

# ✅ Streamlit UI
st.title("AI-Powered Legal HelpDesk")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    chunks = chunk_pdf(uploaded_file)
    st.write(f"Extracted {len(chunks)} chunks from {uploaded_file.name}")
    
    store_vectors(chunks, uploaded_file.name)
    st.success("PDF processed and stored in Pinecone successfully!")
