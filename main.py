import streamlit as st
import pinecone
import PyPDF2
import os
import re
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index_name = "helpdesk"


# Load Hugging Face Model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Regex patterns for Chapters & Articles
chapter_pattern = r'^(Chapter (\d+|[A-Za-z]+)):.*$'
article_pattern = r'^(Article (\d+|[A-Za-z]+)):.*$'

def extract_text_from_pdf(pdf_path):
    """Extracts structured text (chapters & articles) from a PDF."""
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

def store_vectors(pdf_name, chapters, articles):
    """Stores extracted chapters and articles in Pinecone separately."""
    
    # Store chapters
    for i, chapter in enumerate(chapters):
        chapter_vector = model.encode(chapter['content']).tolist()
        index.upsert([
            (f"{pdf_name}-chapter-{i}", chapter_vector, {
                "pdf_name": pdf_name,
                "title": chapter['title'],
                "text": chapter['content'],
                "type": "chapter"
            })
        ])
    
    # Store articles separately
    for i, article in enumerate(articles):
        article_vector = model.encode(article['content']).tolist()
        index.upsert([
            (f"{pdf_name}-article-{i}", article_vector, {
                "pdf_name": pdf_name,
                "chapter": article['chapter'],
                "title": article['title'],
                "text": article['content'],
                "type": "article"
            })
        ])
        
        # Debugging: Print stored article information
        st.write(f"Stored Article {article['title']} (ID: {pdf_name}-article-{i})")

def query_vectors(query, selected_pdfs):
    """Queries Pinecone for the most relevant result from multiple PDFs."""
    query_vector = model.encode(query).tolist()
    
    # Filter by selected PDFs
    filter_query = {"pdf_name": {"$in": selected_pdfs}}

    results = index.query(
        vector=query_vector, 
        top_k=5, 
        include_metadata=True, 
        filter=filter_query
    )
    
    if results and results["matches"]:
        return "\n\n".join([f"**{match['metadata']['title']}**: {match['metadata']['text']}" for match in results["matches"]])
    return "No relevant answer found."

def translate_text(text, target_lang):
    """Translates text using GoogleTranslator."""
    return GoogleTranslator(source="auto", target=target_lang).translate(text)

# Streamlit UI
st.title("AI-Powered Legal HelpDesk")

# PDF Source Selection
pdf_source = st.radio("Select PDF Source", ["Upload from PC", "Choose from Document Storage"], index=1)

if pdf_source == "Upload from PC":
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            temp_pdf_path = os.path.join("/tmp", uploaded_file.name)
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.read())
            
            chapters, articles = extract_text_from_pdf(temp_pdf_path)
            store_vectors(uploaded_file.name, chapters, articles)
        
        st.success("PDFs uploaded and processed successfully!")

# Allow users to query across multiple PDFs
stored_pdfs = ["Basic Law Governance.pdf", "Law of the Consultative Council.pdf", "Law of the Council of Ministers.pdf"]
selected_pdfs = st.multiselect("Select PDFs to Query", stored_pdfs, default=stored_pdfs)

# Language Selection
input_lang = st.radio("Choose Input Language", ["English", "Arabic"], index=0)
response_lang = st.radio("Choose Response Language", ["English", "Arabic"], index=0)

# Query Input
query = st.text_input("Ask a legal question:")

if st.button("Get Answer"):
    if selected_pdfs and query:
        response = query_vectors(query, selected_pdfs)
        if response_lang == "Arabic":
            response = translate_text(response, "ar")
            st.markdown(f"<div dir='rtl' style='text-align: right;'>{response}</div>", unsafe_allow_html=True)
        else:
            st.write(f"**Answer:** {response}")
    else:
        st.warning("Please select PDFs and enter a query.")
