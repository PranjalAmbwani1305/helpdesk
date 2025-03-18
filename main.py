import streamlit as st
import pinecone
import os
import re
import PyPDF2
from sentence_transformers import SentenceTransformer

# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "helpdesk"

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)


# Load Embedding Model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Streamlit UI
st.set_page_config(page_title="Legal HelpDesk - Saudi Arabia", layout="wide")

# Sidebar for PDF Storage
st.sidebar.title("📂 Uploaded PDFs")
if "uploaded_pdfs" not in st.session_state:
    st.session_state["uploaded_pdfs"] = {}

# File Upload Section
st.title("🤖 AI-Powered Legal HelpDesk for Saudi Arabia")
st.write("Upload PDFs and ask legal questions.")

pdf_source = st.radio("Select PDF Source:", ["Upload from PC", "Choose from Document Storage"])

uploaded_file = None
pdf_name = None

if pdf_source == "Upload from PC":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], help="Limit: 200MB per file", accept_multiple_files=False)
    
    if uploaded_file:
        pdf_name = uploaded_file.name
        if pdf_name not in st.session_state["uploaded_pdfs"]:
            st.session_state["uploaded_pdfs"][pdf_name] = uploaded_file

elif pdf_source == "Choose from Document Storage":
    if st.session_state["uploaded_pdfs"]:
        pdf_name = st.selectbox("Select a previously uploaded PDF:", list(st.session_state["uploaded_pdfs"].keys()))
        uploaded_file = st.session_state["uploaded_pdfs"][pdf_name]
    else:
        st.warning("No PDFs uploaded yet.")

# Language Selection
st.subheader("🌍 Choose Languages")
input_language = st.radio("Choose Input Language", ["English", "Arabic"], horizontal=True)
response_language = st.radio("Choose Response Language", ["English", "Arabic"], horizontal=True)

# Function to Extract Articles
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
            article_content.append(line)

        elif current_article:
            article_content.append(line)

    # Store the last article
    if current_article and article_content:
        articles.append({
            "article_number": current_article,
            "chapter_number": current_chapter,
            "text": " ".join(article_content).strip(),
            "pdf_name": pdf_name
        })

    return articles

# Extract Text and Store in Pinecone
if uploaded_file:
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")

    articles = extract_articles(text, pdf_name)

    vectors = []
    for article in articles:
        article_id = f"{pdf_name}-article-{article['article_number']}"
        embedding = model.encode(article["text"]).tolist()

        vectors.append({
            "id": article_id,
            "values": embedding,
            "metadata": {
                "article_number": article["article_number"],
                "chapter_number": article["chapter_number"],
                "pdf_name": pdf_name,
                "text": article["text"],
                "type": "article"
            }
        })

    # Store in Pinecone
    if vectors:
        index.upsert(vectors)
        st.success(f"✅ Stored {len(vectors)} articles in Pinecone.")

# Search Query
st.subheader("🔍 Ask a Legal Question")
query = st.text_area("Enter your question (English or Arabic):")

if query:
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
        st.warning("No relevant legal articles found.")
