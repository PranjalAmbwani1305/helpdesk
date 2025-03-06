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
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# Check if index exists, else create it
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # Ensure this matches your embedding model output
        metric="cosine"
    )

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
    current_chapter, current_chapter_number, current_chapter_content = "Uncategorized", None, []
    current_article, current_article_number, current_article_content = None, None, []

    paragraphs = text.split('\n')

    for para in paragraphs:
        para = para.strip()

        # Detect Chapters
        chapter_match = re.match(chapter_pattern, para)
        if chapter_match:
            if current_chapter != "Uncategorized":
                chapters.append({
                    'title': current_chapter,
                    'chapter_number': current_chapter_number,
                    'content': ' '.join(current_chapter_content)
                })
            current_chapter = para
            current_chapter_number = chapter_match.group(2)  
            current_chapter_content = []
        
        # Detect Articles
        article_match = re.match(article_pattern, para)
        if article_match:
            if current_article:
                articles.append({
                    'chapter': current_chapter,
                    'title': current_article,
                    'article_number': current_article_number,
                    'content': ' '.join(current_article_content)
                })
            current_article = para
            current_article_number = article_match.group(2)
            current_article_content = []
        
        # Add content to current section
        else:
            if current_article:
                current_article_content.append(para)
            else:
                current_chapter_content.append(para)

    # Append last detected sections
    if current_article:
        articles.append({
            'chapter': current_chapter,
            'title': current_article,
            'article_number': current_article_number,
            'content': ' '.join(current_article_content)
        })
    if current_chapter and current_chapter != "Uncategorized":
        chapters.append({
            'title': current_chapter,
            'chapter_number': current_chapter_number,
            'content': ' '.join(current_chapter_content)
        })

    return chapters, articles

# Function to store Chapters & Articles as vectors in Pinecone
def store_vectors(chapters, articles, pdf_name):
    for i, chapter in enumerate(chapters):
        chapter_vector = model.encode(chapter['content']).tolist()
        index.upsert([
            (f"{pdf_name}-chapter-{i}", chapter_vector, {
                "pdf_name": pdf_name,
                "chapter_number": chapter['chapter_number'],
                "title": chapter["title"],
                "text": chapter['content'],
                "type": "chapter"
            })
        ])

    for i, article in enumerate(articles):
        article_vector = model.encode(article['content']).tolist()
        index.upsert([
            (f"{pdf_name}-article-{i}", article_vector, {
                "pdf_name": pdf_name,
                "chapter_number": article['chapter'],
                "article_number": article['article_number'],
                "text": article['content'],
                "type": "article"
            })
        ])

# Function to query vectors from Pinecone
def query_vectors(query, selected_pdf):
    try:
        vector = model.encode(query).tolist()
        
        # Detect if the user is asking about a specific chapter
        chapter_match = re.search(r'chapter\s*(\d+|one|two|three|four|five)', query, re.IGNORECASE)
        filter_query = {"pdf_name": {"$eq":
