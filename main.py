import streamlit as st
import pinecone
import PyPDF2
import os
import re
from sentence_transformers import SentenceTransformer

# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "helpdesk"


pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)


# Load Hugging Face Model for Embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Streamlit UI Layout
st.set_page_config(page_title="PDF Article Extractor", layout="wide")

st.title("📜 PDF Article Extractor & Pinecone Storage")
st.write("Upload multiple PDFs, extract articles, and store them in Pinecone.")

# File Uploader
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

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

def extract_articles(text, pdf_name):
    """Extracts articles and their metadata from the PDF."""
    articles = []
    article_pattern = re.compile(r"\bArticle\s+(\d+)\b", re.IGNORECASE)
    chapter_pattern = re.compile(r"\bChapter\s+\d+:\s*(.+)", re.IGNORECASE)

    lines = text.split("\n")
    current_article = None
    current_chapter = "Unknown Chapter"
    article_content = []

    for line in lines:
        line = line.strip()

        # Detect Chapter
        chapter_match = chapter_pattern.search(line)
        if chapter_match:
            current_chapter = chapter_match.group(1).strip()

        # Detect Article
        article_match = article_pattern.match(line)
        if article_match:
            if current_article and article_content:
                articles.append({
                    "article_number": current_article,
                    "chapter_number": current_chapter,
                    "text": " ".join(article_content).strip(),
                    "pdf_name": pdf_name
                })
                article_content = []

            current_article = article_match.group(1)
            article_content.append(line)  # Start new article content

        elif current_article:
            article_content.append(line)  # Append content to the current article

    # Store the last article
    if current_article and article_content:
        articles.append({
            "article_number": current_article,
            "chapter_number": current_chapter,
            "text": " ".join(article_content).strip(),
            "pdf_name": pdf_name
        })

    return articles

def store_articles_in_pinecone(articles):
    """Stores articles in Pinecone with proper embeddings."""
    vectors = []
    for article in articles:
        article_id = f"{article['pdf_name']}-article-{article['article_number']}"
        embedding = model.encode(article["text"]).tolist()

        vectors.append({
            "id": article_id,
            "values": embedding,
            "metadata": {
                "article_number": article["article_number"],
                "chapter_number": article["chapter_number"],
                "pdf_name": article["pdf_name"],
                "text": article["text"],
                "type": "article"
            }
        })

    # Store in Pinecone
    if vectors:
        index.upsert(vectors)
        st.success(f"✅ Stored {len(vectors)} articles in Pinecone.")

def process_pdfs_and_store(uploaded_files):
    """Extracts articles from multiple PDFs and stores them in Pinecone."""
    all_articles = []
    for pdf_file in uploaded_files:
        pdf_name = os.path.basename(pdf_file.name)
        text = extract_text_from_pdf(pdf_file)
        articles = extract_articles(text, pdf_name)
        all_articles.extend(articles)
    store_articles_in_pinecone(all_articles)

# Process PDFs
if uploaded_files:
    with st.spinner("Processing PDFs..."):
        process_pdfs_and_store(uploaded_files)

# Search UI
st.subheader("🔍 Search Articles")
query = st.text_input("Enter search query")

if query:
    with st.spinner("Searching..."):
        query_embedding = model.encode(query).tolist()
        results = index.query(queries=[query_embedding], top_k=5, include_metadata=True)

        if results and results['results']:
            for match in results['results'][0]['matches']:
                metadata = match['metadata']
                st.markdown(f"### **Article {metadata['article_number']}**")
                st.markdown(f"📂 **PDF:** {metadata['pdf_name']}")
                st.markdown(f"📖 **Chapter:** {metadata['chapter_number']}")
                st.markdown(f"✍ **Text:** {metadata['text']}")
                st.write("---")
        else:
            st.warning("No matching articles found.")
