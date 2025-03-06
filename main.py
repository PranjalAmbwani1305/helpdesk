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
        
        # Check if user queries a specific chapter
        chapter_match = re.search(r'chapter\s*(\d+)', query, re.IGNORECASE)
        filter_query = {"pdf_name": {"$eq": selected_pdf}}
        
        if chapter_match:
            chapter_number = chapter_match.group(1)
            filter_query["chapter_number"] = {"$eq": chapter_number}

        results = index.query(vector=vector, top_k=5, include_metadata=True, filter=filter_query)
        
        if results["matches"]:
            matched_texts = []
            
            for match in results["matches"]:
                section_type = match["metadata"]["type"]
                section_text = match["metadata"]["text"]
                
                if section_type == "chapter":
                    matched_texts.append(f"**Chapter {match['metadata']['chapter_number']}:** {section_text}")
                elif section_type == "article":
                    chapter_name = match["metadata"].get("chapter_number", "Uncategorized")
                    article_number = match["metadata"].get("article_number", "Unknown")
                    matched_texts.append(f"**Chapter {chapter_name} - Article {article_number}:**\n{section_text}")
            
            return "\n\n".join(matched_texts)
        else:
            return "No relevant information found in the selected document."
    except Exception as e:
        return f"Error during query: {e}"

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>AI-Powered Legal HelpDesk</h1>", unsafe_allow_html=True)

# PDF Source Selection
pdf_source = st.radio("Select PDF Source", ("Upload from PC", "Choose from Document Storage"))

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

        st.success("PDF uploaded and processed successfully!")
else:
    st.info("Document Storage feature is currently unavailable.")

# Language Selection
st.subheader("Choose Input Language")
input_language = st.selectbox("Select Input Language", ("English", "Arabic"))

st.subheader("Choose Response Language")
response_language = st.selectbox("Select Response Language", ("English", "Arabic"))

# Query Input and Processing
query = st.text_input("Ask a question (in English or Arabic):")

if st.button("Get Answer"):
    if query and uploaded_file:
        response = query_vectors(query, uploaded_file.name)
        st.write(f"**Answer:** {response}")
    else:
        st.warning("Please upload a PDF and enter a query.")
