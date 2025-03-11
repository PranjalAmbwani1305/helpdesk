import streamlit as st
import fitz  # PyMuPDF
import re
from sentence_transformers import SentenceTransformer
import os
from pinecone import Pinecone

# Load Pinecone API Key
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")  # Use your actual key here

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index_name = "helpdesk"
index = pc.Index(index_name)

# Load embedding model
model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

# Function to extract articles and chapters from PDFs
def extract_text_from_pdf(pdf_file):
    """Extracts text from PDF and organizes it into chapters & articles."""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""

    for page in doc:
        text += page.get_text("text") + "\n"

    chapters = []
    articles = []
    chapter_text = ""
    current_chapter = ""

    lines = text.split("\n")
    for line in lines:
        line = line.strip()

        # Detect chapters (e.g., "Chapter One: General Principles")
        if re.match(r"^Chapter\s+[A-Za-z0-9]+:", line):
            if chapter_text:
                chapters.append({"title": current_chapter, "content": chapter_text.strip()})
            current_chapter = line
            chapter_text = ""

        # Detect articles (e.g., "Article 1")
        elif re.match(r"^Article\s+\d+", line):
            if chapter_text:
                articles.append({"chapter": current_chapter, "title": line, "content": chapter_text.strip()})
            chapter_text = ""

        chapter_text += line + " "

    # Store last chapter & article
    if chapter_text:
        articles.append({"chapter": current_chapter, "title": line, "content": chapter_text.strip()})

    return chapters, articles

# Function to store extracted text as vector embeddings in Pinecone
def store_vectors(chapters, articles, pdf_name):
    """Stores extracted chapters & articles in Pinecone."""
    for i, chapter in enumerate(chapters):
        chapter_vector = model.encode(chapter["content"]).tolist()
        index.upsert([(
            f"{pdf_name}-chapter-{i}", chapter_vector, 
            {"pdf_name": pdf_name, "text": chapter["content"], "type": "chapter", "title": chapter["title"]}
        )])

    for i, article in enumerate(articles):
        article_vector = model.encode(article["content"]).tolist()
        
        # Correct article title formatting
        article_number = article["title"].replace("Article ", "").strip()
        formatted_title = f"Article {article_number}"

        index.upsert([(
            f"{pdf_name}-article-{i}", article_vector, 
            {"pdf_name": pdf_name, "chapter": article["chapter"], "text": article["content"], "type": "article", "title": formatted_title}
        )])

# Function to query Pinecone
def query_vectors(query, selected_pdf):
    """Queries Pinecone for the most relevant result, prioritizing article matches."""
    query_vector = model.encode(query).tolist()
    
    # Check if user is specifically asking for an article (e.g., "Article 1")
    article_match = re.search(r'Article (\d+|[A-Za-z]+)', query, re.IGNORECASE)
    
    if article_match:
        article_number = article_match.group(1).strip()
        formatted_title = f"Article {article_number}"

        # Search for exact article title match
        results = index.query(vector=query_vector, top_k=5, include_metadata=True)
        for match in results["matches"]:
            if match["metadata"]["title"] == formatted_title and match["metadata"]["pdf_name"] == selected_pdf:
                return match["metadata"]["text"]

    # If no article match, return general best match
    results = index.query(vector=query_vector, top_k=1, include_metadata=True)
    if results["matches"]:
        return results["matches"][0]["metadata"]["text"]
    
    return "No matching article found."

# Streamlit UI
st.title("📜 AI-Powered Legal Helpdesk")
st.sidebar.header("Upload Legal Documents")

uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
if uploaded_file:
    chapters, articles = extract_text_from_pdf(uploaded_file)
    store_vectors(chapters, articles, uploaded_file.name)
    st.sidebar.success("📌 PDF processed successfully!")

st.header("Ask a Legal Question")
query = st.text_input("Enter your legal question here:")
if st.button("Search"):
    if uploaded_file:
        answer = query_vectors(query, uploaded_file.name)
        st.write("### 📖 Answer")
        st.write(answer)
    else:
        st.warning("⚠️ Please upload a legal document first.")
