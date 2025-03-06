import streamlit as st
import pinecone
import PyPDF2
import os
import re
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator

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
    
    chapters, articles = [], []
    current_chapter, current_chapter_content = "Uncategorized", []
    current_article, current_article_content = None, []

    paragraphs = text.split('\n')

    for para in paragraphs:
        para = para.strip()

        # Detect Chapters
        if re.match(chapter_pattern, para):
            if current_chapter != "Uncategorized":
                chapters.append({'title': current_chapter, 'content': ' '.join(current_chapter_content)})
            current_chapter = para
            current_chapter_content = []
        
        # Detect Articles
        article_match = re.match(article_pattern, para)
        if article_match:
            if current_article:
                articles.append({
                    'chapter': current_chapter, 
                    'title': current_article, 
                    'article_number': article_match.group(2) if article_match else "Unknown",
                    'content': ' '.join(current_article_content)
                })
            current_article = article_match.group(1)
            current_article_content = []
        
        # Add content to current section
        else:
            if current_article:
                current_article_content.append(para)
            else:
                current_chapter_content.append(para)

    # Append last detected sections
    if current_article:
        articles.append({'chapter': current_chapter, 'title': current_article, 'content': ' '.join(current_article_content)})
    if current_chapter and current_chapter != "Uncategorized":
        chapters.append({'title': current_chapter, 'content': ' '.join(current_chapter_content)})

    return chapters, articles

# Function to store Chapters & Articles as vectors in Pinecone
def store_vectors(chapters, articles, pdf_name):
    for i, chapter in enumerate(chapters):
        chapter_content = chapter.get("content", "").strip()
        if not chapter_content:
            print(f"⚠️ Skipping empty chapter: {chapter}")
            continue

        chapter_vector = model.encode(chapter_content).tolist()
        index.upsert([
            (f"{pdf_name}-chapter-{i}", chapter_vector, {"pdf_name": pdf_name, "text": chapter_content, "type": "chapter"})
        ])

    for i, article in enumerate(articles):
        article_content = article.get("content", "").strip()
        if not article_content:
            print(f"⚠️ Skipping empty article: {article}")
            continue

        article_vector = model.encode(article_content).tolist()
        index.upsert([
            (f"{pdf_name}-article-{i}", article_vector, {
                "pdf_name": pdf_name,
                "chapter": article.get("chapter", "Unknown"),
                "article_number": article.get("article_number", "Unknown"),
                "text": article_content,
                "type": "article"
            })
        ])

# Function to query vectors from Pinecone
def query_vectors(query, selected_pdf):
    try:
        vector = model.encode(query).tolist()
        results = index.query(vector=vector, top_k=5, include_metadata=True, filter={"pdf_name": {"$eq": selected_pdf}})
        
        if results["matches"]:
            matched_texts = []
            
            for match in results["matches"]:
                metadata = match.get("metadata", {})
                section_type = metadata.get("type", "Unknown")
                section_text = metadata.get("text", "No text found")

                if section_type == "chapter":
                    matched_texts.append(f"**Chapter:** {section_text}")
                elif section_type == "article":
                    chapter_name = metadata.get("chapter", "Uncategorized")
                    article_number = metadata.get("article_number", "Unknown")
                    matched_texts.append(f"**{chapter_name} - Article {article_number}**\n{section_text}")

            return "\n\n".join(matched_texts)
        else:
            return "No relevant information found in the selected document."
    except Exception as e:
        return f"Error during query: {e}"

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>AI-Powered Legal HelpDesk</h1>", unsafe_allow_html=True)

# PDF Source Selection
pdf_source = st.radio("Select PDF Source", ("Upload from PC", "Choose from Document Storage"))

selected_pdf = None

if pdf_source == "Upload from PC":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        temp_pdf_path = f"temp_{uploaded_file.name}"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Process PDF to extract chapters & articles
        chapters, articles = process_pdf(temp_pdf_path)

        # Store extracted sections in Pinecone
        store_vectors(chapters, articles, uploaded_file.name)

        selected_pdf = uploaded_file.name
        st.success("PDF uploaded and processed successfully!")

else:
    st.info("Document Storage feature is currently unavailable.")

# Language Selection
input_language = st.selectbox("Choose Input Language", ("English", "Arabic"))
response_language = st.selectbox("Choose Response Language", ("English", "Arabic"))

# Query Input and Processing
query = st.text_input("Ask a legal question:")

if st.button("Get Answer"):
    if selected_pdf and query:
        detected_lang = GoogleTranslator(source="auto", target="en").translate(query)
        
        response = query_vectors(detected_lang, selected_pdf)

        if response_language == "Arabic":
            response = GoogleTranslator(source="auto", target="ar").translate(response)
            st.markdown(f"<div dir='rtl' style='text-align: right;'>{response}</div>", unsafe_allow_html=True)
        else:
            st.write(f"**Answer:** {response}")
    else:
        st.warning("Please enter a query and select a PDF.")
