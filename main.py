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

# Regex patterns
chapter_pattern = r'^(Chapter (\d+|[A-Za-z]+)):.*$'
article_pattern = r'^(Article (\d+|[A-Za-z]+)):.*$'

def extract_text_from_pdf(pdf_path):
    """Extracts structured text from the PDF."""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
    chapters, articles = [], []
    current_chapter = "Uncategorized"
    current_article, current_article_content = None, []
    
    for para in text.split('\n'):
        para = para.strip()
        
        if re.match(chapter_pattern, para):
            current_chapter = para
        
        article_match = re.match(article_pattern, para)
        if article_match:
            if current_article:
                articles.append({'chapter': current_chapter, 'title': current_article, 'content': ' '.join(current_article_content)})
            current_article = article_match.group(1)
            current_article_content = []
        else:
            if current_article:
                current_article_content.append(para)
    
    if current_article:
        articles.append({'chapter': current_chapter, 'title': current_article, 'content': ' '.join(current_article_content)})
    return articles


def store_vectors(articles, pdf_name):
    """Stores extracted articles in Pinecone."""
    for i, article in enumerate(articles):
        article_vector = model.encode(article['content']).tolist()
        index.upsert([
            (f"{pdf_name}-article-{i}", article_vector, {
                "pdf_name": pdf_name, "chapter": article['chapter'],
                "text": article['content'], "type": "article", "title": article['title']
            })
        ])


def query_vectors(query, selected_pdf):
    """Queries Pinecone for the most relevant result."""
    query_vector = model.encode(query).tolist()
    
    # Handling Chapter Queries
    chapter_match = re.search(r'Chapter (\d+|[A-Za-z]+)', query, re.IGNORECASE)
    if chapter_match:
        chapter_number = chapter_match.group(1)
        results = index.query(
            vector=query_vector,
            top_k=5,
            include_metadata=True,
            filter={"pdf_name": {"$eq": selected_pdf}, "chapter": {"$regex": f"Chapter {chapter_number}.*"}}
        )
        if results and results["matches"]:
            return "\n\n".join([match["metadata"]["text"] for match in results["matches"]])
    
    # Handling Article Queries
    article_match = re.search(r'Article (\d+|[A-Za-z]+)', query, re.IGNORECASE)
    if article_match:
        article_number = article_match.group(1)
        results = index.query(
            vector=query_vector,
            top_k=1,
            include_metadata=True,
            filter={"pdf_name": {"$eq": selected_pdf}, "type": {"$eq": "article"}, "title": {"$eq": f"Article {article_number}"}}
        )
        if results and results["matches"]:
            return results["matches"][0]["metadata"]["text"]
    
    return "No relevant answer found."

def translate_text(text, target_lang):
    """Translates text."""
    return GoogleTranslator(source="auto", target=target_lang).translate(text)

# Streamlit UI
st.set_page_config(page_title="Legal HelpDesk", layout="wide")
st.title("📜 AI-Powered Legal HelpDesk")
st.write("Ask legal questions and retrieve specific articles & chapters instantly.")

# PDF Source Selection
pdf_source = st.radio("Select PDF Source", ["Upload from PC", "Choose from Document Storage"], index=1)
selected_pdf = None

if pdf_source == "Upload from PC":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file is not None:
        temp_pdf_path = os.path.join("/tmp", uploaded_file.name)
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.read())
        
        articles = extract_text_from_pdf(temp_pdf_path)
        store_vectors(articles, uploaded_file.name)
        selected_pdf = uploaded_file.name
        st.success("PDF uploaded and processed successfully!")
else:
    stored_pdfs = ["Basic Law Governance.pdf", "Law of the Consultative Council.pdf", "Law of the Council of Ministers.pdf"]
    selected_pdf = st.selectbox("Select a PDF", stored_pdfs)

# Language Selection
input_lang = st.radio("Choose Input Language", ["English", "Arabic"], index=0)
response_lang = st.radio("Choose Response Language", ["English", "Arabic"], index=0)

# Query Input
query = st.text_input("Ask a legal question:")
if st.button("🔍 Get Answer"):
    if selected_pdf and query:
        response = query_vectors(query, selected_pdf)
        if response_lang == "Arabic":
            response = translate_text(response, "ar")
            st.markdown(f"<div dir='rtl' style='text-align: right;'>{response}</div>", unsafe_allow_html=True)
        else:
            st.write(f"**Answer:** {response}")
    else:
        st.warning("Please upload a PDF and enter a query.")
