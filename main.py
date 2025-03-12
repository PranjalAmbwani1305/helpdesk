import streamlit as st
import pinecone
import PyPDF2
import os
import re
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
from pinecone import Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "helpdesk"
index = pc.Index(index_name)

# Load Hugging Face Model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Regex patterns for Chapters & Articles
chapter_pattern = r'^(Chapter (\d+|[A-Za-z]+)):.*$'
article_pattern = r'^(Article (\d+|[A-Za-z]+)):.*$'

def extract_text_from_pdf(pdf_path):
    """Extracts and structures text from the PDF."""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
    chapters, articles = [], []
    current_chapter, current_chapter_content = "Uncategorized", []
    current_article, current_article_content = None, []
    paragraphs = text.split('\n')
    
    for para in paragraphs:
        para = para.strip()
        
        if re.match(chapter_pattern, para):
            if current_chapter != "Uncategorized":
                chapters.append({'title': current_chapter, 'content': ' '.join(current_chapter_content)})
            current_chapter = para
            current_chapter_content = []
        
        article_match = re.match(article_pattern, para)
        if article_match:
            if current_article:
                articles.append({'chapter': current_chapter, 'title': current_article, 'content': ' '.join(current_article_content)})
            current_article = article_match.group(1)
            current_article_content = []
        else:
            if current_article:
                current_article_content.append(para)
            else:
                current_chapter_content.append(para)

    if current_article:
        articles.append({'chapter': current_chapter, 'title': current_article, 'content': ' '.join(current_article_content)})
    if current_chapter and current_chapter != "Uncategorized":
        chapters.append({'title': current_chapter, 'content': ' '.join(current_chapter_content)})
    
    return chapters, articles

def store_vectors(chapters, articles, pdf_name):
    """Stores extracted chapters and articles in Pinecone."""
    namespace = pdf_name.replace(" ", "_").lower()  # Unique namespace for each PDF

    batch = []
    for i, chapter in enumerate(chapters):
        chapter_vector = model.encode(chapter['content']).tolist()
        batch.append((
            f"{pdf_name}-chapter-{i}", chapter_vector, 
            {"pdf_name": pdf_name, "text": chapter['content'], "type": "chapter"}
        ))
    
    for i, article in enumerate(articles):
        article_number_match = re.search(r'Article (\d+|[A-Za-z]+)', article['title'], re.IGNORECASE)
        article_number = article_number_match.group(1) if article_number_match else str(i)
        article_vector = model.encode(article['content']).tolist()
        batch.append((
            f"{pdf_name}-article-{article_number}", article_vector, 
            {"pdf_name": pdf_name, "chapter": article['chapter'], "text": article['content'], "type": "article", "title": article['title']}
        ))

    # Insert batch into Pinecone
    if batch:
        index.upsert(batch, namespace=namespace)
        st.success(f"PDF '{pdf_name}' stored in Pinecone successfully!")

def get_stored_pdfs():
    """Retrieves a list of stored PDFs from Pinecone."""
    index_stats = index.describe_index_stats()
    namespaces = index_stats.get("namespaces", {})
    stored_pdfs = list(namespaces.keys())  # List all stored PDF namespaces
    return stored_pdfs

def query_vectors(query, selected_pdf):
    """Queries Pinecone for the most relevant result."""
    query_vector = model.encode(query).tolist()
    namespace = selected_pdf.replace(" ", "_").lower()

    results = index.query(
        vector=query_vector,
        top_k=5, 
        include_metadata=True, 
        namespace=namespace
    )
    
    if results and results["matches"]:
        return "\n\n".join([match["metadata"]["text"] for match in results["matches"]])
    
    return "No relevant answer found."

def translate_text(text, target_lang):
    """Translates text using GoogleTranslator."""
    return GoogleTranslator(source="auto", target=target_lang).translate(text)

# Streamlit UI
st.title("📜 AI-Powered Legal HelpDesk")

# Sidebar: List stored PDFs
st.sidebar.header("📂 Stored PDFs")
stored_pdfs = get_stored_pdfs()
selected_pdf = None

# PDF Source Selection
pdf_source = st.radio("Select PDF Source", ["Upload from PC", "Choose from Document Storage"], index=1)

if pdf_source == "Upload from PC":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file is not None:
        temp_pdf_path = os.path.join("/tmp", uploaded_file.name)
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.read())
        
        chapters, articles = extract_text_from_pdf(temp_pdf_path)
        store_vectors(chapters, articles, uploaded_file.name)
        st.rerun()  # Refresh page to update the sidebar

else:
    if stored_pdfs:
        selected_pdf = st.sidebar.radio("Select a PDF", stored_pdfs)
    else:
        st.sidebar.warning("No PDFs found in storage.")

# Language Selection
input_lang = st.radio("Choose Input Language", ["English", "Arabic"], index=0)
response_lang = st.radio("Choose Response Language", ["English", "Arabic"], index=0)

# Query Input
query = st.text_input("Ask a legal question:")

if st.button("Get Answer"):
    if selected_pdf and query:
        response = query_vectors(query, selected_pdf)
        if response_lang == "Arabic":
            response = translate_text(response, "ar")
            st.markdown(f"<div dir='rtl' style='text-align: right;'>{response}</div>", unsafe_allow_html=True)
        else:
            st.write(f"**Answer:** {response}")
    else:
        st.warning("Please select a stored PDF and enter a query.")
