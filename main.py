import streamlit as st
import pinecone
import PyPDF2
import os
import re
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

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

# Regex patterns for Chapters & Articles
chapter_pattern = r'^(Chapter (\d+|[A-Za-z]+)):.*$'
article_pattern = r'^(Article (\d+|[A-Za-z]+)):.*$'

# Function to process PDF and extract Chapters & Articles
def process_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])

    # Remove website-related noise
    text = re.sub(r"PDFmyURL.*?quickly", "", text, flags=re.DOTALL)

    chapters, articles = [], []
    current_chapter, current_chapter_content = "Uncategorized", []
    current_article, current_article_content = None, []

    paragraphs = text.split('\n')

    for para in paragraphs:
        para = para.strip()

        # Detect Chapters
        chapter_match = re.match(chapter_pattern, para)
        if chapter_match:
            if current_chapter != "Uncategorized":
                chapters.append({'title': current_chapter, 'content': ' '.join(current_chapter_content)})
            current_chapter = para
            current_chapter_content = []
            continue
        
        # Detect Articles
        article_match = re.match(article_pattern, para)
        if article_match:
            if current_article:
                articles.append({'chapter': current_chapter, 'title': current_article, 'article_number': article_match.group(2), 'content': ' '.join(current_article_content)})
            current_article = para
            current_article_content = []
            continue
        
        # Add content
        if current_article:
            current_article_content.append(para)
        else:
            current_chapter_content.append(para)

    # Append last detected sections
    if current_article:
        articles.append({'chapter': current_chapter, 'title': current_article, 'article_number': article_match.group(2), 'content': ' '.join(current_article_content)})
    if current_chapter and current_chapter != "Uncategorized":
        chapters.append({'title': current_chapter, 'content': ' '.join(current_chapter_content)})

    return chapters, articles

# Function to store Chapters & Articles as vectors in Pinecone
def store_vectors(chapters, articles, pdf_name):
    if not chapters and not articles:
        st.error("No data extracted from the PDF. Please check the document format.")
        return
    
    st.write("📌 Storing the first extracted record for debugging:")

    if chapters:
        chapter_vector = model.encode(chapters[0]['content']).tolist()
        st.write(f"Storing Chapter: {chapters[0]['title']} - {chapters[0]['content'][:100]}...")
        index.upsert([
            (f"{pdf_name}-chapter-0", chapter_vector, {"pdf_name": pdf_name, "text": chapters[0]['content'], "type": "chapter"})
        ])

    if articles:
        article_vector = model.encode(articles[0]['content']).tolist()
        st.write(f"Storing Article: {articles[0]['title']} - {articles[0]['content'][:100]}...")
        index.upsert([
            (f"{pdf_name}-article-0", article_vector, {
                "pdf_name": pdf_name, 
                "chapter": articles[0]['chapter'], 
                "article_number": articles[0]['article_number'],  
                "text": articles[0]['content'], 
                "type": "article"
            })
        ])

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>AI-Powered Legal HelpDesk</h1>", unsafe_allow_html=True)

# PDF Source Selection
pdf_source = st.radio("Select PDF Source", ("Upload from PC", "Choose from Document Storage"))

selected_file = None  

if pdf_source == "Upload from PC":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        temp_pdf_path = f"temp_{uploaded_file.name}"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.read())
        
        chapters, articles = process_pdf(temp_pdf_path)
        store_vectors(chapters, articles, uploaded_file.name)

        st.success("PDF uploaded and processed successfully!")
        selected_file = uploaded_file.name  

elif pdf_source == "Choose from Document Storage":
    storage_folder = "document_storage"
    stored_pdfs = [file for file in os.listdir(storage_folder) if file.endswith(".pdf")] if os.path.exists(storage_folder) else []
    
    if stored_pdfs:
        selected_file = st.selectbox("Select a stored document:", stored_pdfs)
    else:
        st.warning("No documents available in storage.")

if selected_file:
    st.success(f"Selected document: {selected_file}")
