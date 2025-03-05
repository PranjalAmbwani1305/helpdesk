import streamlit as st
import pinecone
import PyPDF2
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import re

# Load environment variables for Pinecone API key
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
index_name = "helpdesk"

# Initialize Pinecone
from pinecone import Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(index_name)

# Initialize sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to process PDF and extract chapters, articles, and content
def process_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
    sections = []
    current_chapter = None
    current_article = None
    current_content = []

    chapter_pattern = r'^(Chapter [A-Za-z\d]+): (.*)$'  # Detect "Chapter One: Name"
    article_pattern = r'^(Article \d+):'  # Detect "Article 1:"

    paragraphs = text.split('\n')

    for para in paragraphs:
        if re.match(chapter_pattern, para.strip()):
            if current_article:
                sections.append({'chapter': current_chapter, 'article': current_article, 'content': ' '.join(current_content)})
            current_chapter = para.strip()  # Store Chapter
            current_article = None
            current_content = []
        elif re.match(article_pattern, para.strip()):
            if current_article:
                sections.append({'chapter': current_chapter, 'article': current_article, 'content': ' '.join(current_content)})
            current_article = para.strip()  # Store Article
            current_content = []
        else:
            current_content.append(para.strip())

    if current_article:
        sections.append({'chapter': current_chapter, 'article': current_article, 'content': ' '.join(current_content)})

    return sections

# Function to store vectors of chapter, article, and content in Pinecone
def store_vectors(sections, pdf_name):
    for i, section in enumerate(sections):
        chapter = section.get('chapter', 'Unknown Chapter')
        article = section.get('article', 'Unknown Article')
        content = section.get('content', '')

        # Create vectors
        chapter_vector = model.encode(chapter).tolist()
        article_vector = model.encode(article).tolist()
        content_vector = model.encode(content).tolist()

        # Store chapter, article, and content in Pinecone
        index.upsert([
            (f"{pdf_name}-chapter-{i}", chapter_vector, {"pdf_name": pdf_name, "chapter": chapter, "article": "", "text": chapter, "type": "chapter"}),
            (f"{pdf_name}-article-{i}", article_vector, {"pdf_name": pdf_name, "chapter": chapter, "article": article, "text": article, "type": "article"}),
            (f"{pdf_name}-content-{i}", content_vector, {"pdf_name": pdf_name, "chapter": chapter, "article": article, "text": content, "type": "content"})
        ])

# Function to query Pinecone and fetch relevant chapters/articles
def search_pinecone(query):
    query_vector = model.encode(query).tolist()
    
    # Perform similarity search
    search_results = index.query(query_vector, top_k=10, include_metadata=True)

    retrieved_texts = []
    
    for match in search_results["matches"]:
        metadata = match["metadata"]
        pdf_name = metadata.get("pdf_name", "Unknown PDF")
        chapter = metadata.get("chapter", "Unknown Chapter")
        article = metadata.get("article", "Unknown Article")
        text = metadata.get("text", "")

        retrieved_texts.append(f"📖 **{pdf_name}**\n📂 **{chapter}**\n📝 **{article}**\n{text}")

    if retrieved_texts:
        return "\n\n".join(retrieved_texts)
    else:
        return "No relevant information found in the selected document."

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>AI-Powered Legal HelpDesk</h1>", unsafe_allow_html=True)

# Sidebar - List uploaded PDFs from Pinecone
st.sidebar.header("Uploaded PDFs")
try:
    index_stats = index.describe_index_stats()
    stored_pdfs = list(set([match["metadata"]["pdf_name"] for match in index_stats["namespaces"].get("", {}).get("metadata", [])]))
    selected_pdf = st.sidebar.selectbox("Select a PDF", stored_pdfs if stored_pdfs else ["No PDFs found"])
except:
    selected_pdf = None

# PDF Upload Section
pdf_source = st.radio("Select PDF Source", ("Upload from PC", "Choose from Document Storage"))

if pdf_source == "Upload from PC":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        temp_pdf_path = f"temp_{uploaded_file.name}"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.read())
        
        sections = process_pdf(temp_pdf_path)
        store_vectors(sections, uploaded_file.name)
        st.success("PDF uploaded and processed!")

# Query Section
query = st.text_input("Ask a legal question:")

if st.button("Get Answer"):
    if query and selected_pdf:
        response = search_pinecone(query)
        st.write(f"**Answer:** {response}")
    else:
        st.warning("Please select a PDF and enter a query.")
