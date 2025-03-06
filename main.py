import streamlit as st
import os
import PyPDF2
import torch
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "helpdesk"

# Create index if not exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # Hugging Face MiniLM has 384 dimensions
        metric="cosine"
    )

index = pc.Index(index_name)

# Load Hugging Face model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Function to get embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

# Function to process PDFs into articles and chapters
def process_pdf(pdf_path):
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        
        # Splitting text into articles and chapters
        articles = text.split("Article")
        chapters = text.split("Chapter")

        article_chunks = [f"Article {i}" + art.strip() for i, art in enumerate(articles) if art.strip()]
        chapter_chunks = [f"Chapter {i}" + ch.strip() for i, ch in enumerate(chapters) if ch.strip()]

        return article_chunks, chapter_chunks
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return [], []

# Store embeddings in Pinecone
def store_vectors(articles, chapters, pdf_name):
    try:
        vectors = []
        for i, article in enumerate(articles):
            vector = get_embedding(article)
            doc_id = f"{pdf_name}-article-{i}"
            metadata = {"pdf_name": pdf_name, "type": "article", "content": article}
            vectors.append((doc_id, vector, metadata))

        for i, chapter in enumerate(chapters):
            vector = get_embedding(chapter)
            doc_id = f"{pdf_name}-chapter-{i}"
            metadata = {"pdf_name": pdf_name, "type": "chapter", "content": chapter}
            vectors.append((doc_id, vector, metadata))

        print(f"✅ Storing {len(vectors)} vectors in Pinecone...")
        index.upsert(vectors)
    except Exception as e:
        print(f"❌ Error storing vectors: {str(e)}")

# Query Pinecone
def query_vectors(query, query_type="article"):
    vector = get_embedding(query)
    results = index.query(vector=vector, top_k=5, include_metadata=True, filter={"type": query_type})

    if results["matches"]:
        return [match["metadata"]["content"] for match in results["matches"]]
    else:
        return ["No relevant information found."]

# Streamlit UI
st.title("📜 AI-Powered Legal HelpDesk")

pdf_source = st.radio("Select PDF Source", ["Upload PDF"])

if pdf_source == "Upload PDF":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        pdf_name = uploaded_file.name
        temp_pdf_path = f"temp_{pdf_name}"

        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        articles, chapters = process_pdf(temp_pdf_path)
        store_vectors(articles, chapters, pdf_name)
        st.success("PDF processed and stored in Pinecone!")

query = st.text_input("Ask a question:")
query_type = st.radio("Search in:", ["Article", "Chapter"], index=0).lower()

if st.button("Search"):
    results = query_vectors(query, query_type)
    for res in results:
        st.write(f"🔹 {res}")
