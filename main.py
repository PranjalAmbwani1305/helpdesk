import streamlit as st
import pinecone
import PyPDF2
import os
import re
from sentence_transformers import SentenceTransformer

# Load Pinecone API Key from Environment Variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "helpdesk"

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Load Hugging Face Model for Embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF file."""
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text.strip()

def extract_articles(text):
    """Extracts only articles from the PDF text."""
    articles = []
    article_pattern = re.compile(r"\bArticle\s+(\d+)\b", re.IGNORECASE)
    chapter_pattern = re.compile(r"\bChapter\s+\d+:\s*(.+)", re.IGNORECASE)

    lines = text.split("\n")
    current_article = None
    current_chapter = "Unknown Chapter"
    article_content = ""

    for line in lines:
        line = line.strip()

        # Detect Chapter
        chapter_match = chapter_pattern.search(line)
        if chapter_match:
            current_chapter = chapter_match.group(1).strip()

        # Detect Article
        article_match = article_pattern.search(line)
        if article_match:
            if current_article and article_content:
                articles.append({
                    "article_number": current_article,
                    "chapter_number": current_chapter,
                    "text": article_content.strip(),
                    "type": "article"
                })
                article_content = ""

            current_article = article_match.group(1)
            article_content = line  # Start new article content

        elif current_article:
            article_content += " " + line  # Append content to the current article

    # Store the last article
    if current_article and article_content:
        articles.append({
            "article_number": current_article,
            "chapter_number": current_chapter,
            "text": article_content.strip(),
            "type": "article"
        })

    return articles

def store_articles(articles, pdf_name):
    """Stores extracted articles in Pinecone."""
    vector_data = []
    
    for i, article in enumerate(articles):
        article_vector = model.encode(article["text"]).tolist()
        vector_data.append((
            f"{pdf_name}-article-{i}", article_vector, {
                "article_number": article["article_number"],
                "chapter_number": article["chapter_number"],
                "pdf_name": pdf_name,
                "text": article["text"],
                "type": article["type"]
            }
        ))

    if vector_data:
        index.upsert(vector_data)
        st.success(f"✅ {len(vector_data)} articles stored in Pinecone from {pdf_name}.")

# Streamlit UI
st.title("📄 AI-Powered Legal HelpDesk for Saudi Arabia")

# Sidebar for Uploaded PDFs
st.sidebar.header("Uploaded PDFs")

# Upload PDF Files
uploaded_files = st.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            pdf_text = extract_text_from_pdf(uploaded_file)
            articles = extract_articles(pdf_text)
            store_articles(articles, uploaded_file.name)

        # Show uploaded file in the sidebar
        st.sidebar.write(f"✅ {uploaded_file.name} stored!")
