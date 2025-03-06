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
chapter_pattern = r'^(Chapter \d+):\s*(.*)$'
article_pattern = r'^(Article \d+):\s*(.*)$'

def process_pdf(pdf_path, pdf_name):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
    articles = []
    current_chapter = None
    current_article = None
    current_article_content = []
    
    for line in text.split("\n"):
        line = line.strip()
        
        chapter_match = re.match(chapter_pattern, line)
        if chapter_match:
            current_chapter = chapter_match.group(2)
        
        article_match = re.match(article_pattern, line)
        if article_match:
            if current_article:
                articles.append({
                    "chapter": current_chapter,
                    "article_number": current_article.split()[1],
                    "text": " ".join(current_article_content),
                    "pdf_name": pdf_name,
                    "type": "article"
                })
            current_article = article_match.group(1)
            current_article_content = []
        else:
            if current_article:
                current_article_content.append(line)
    
    if current_article:
        articles.append({
            "chapter": current_chapter,
            "article_number": current_article.split()[1],
            "text": " ".join(current_article_content),
            "pdf_name": pdf_name,
            "type": "article"
        })
    
    return articles

def store_vectors(articles):
    for i, article in enumerate(articles):
        article_vector = model.encode(article['text']).tolist()
        index.upsert([
            (f"{article['pdf_name']}-article-{article['article_number']}", article_vector, article)
        ])

def query_vectors(query, selected_pdf):
    vector = model.encode(query).tolist()
    filter_query = {"pdf_name": {"$eq": selected_pdf}, "type": {"$eq": "article"}}
    
    results = index.query(vector=vector, top_k=5, include_metadata=True, filter=filter_query)
    
    if results["matches"]:
        return [match["metadata"] for match in results["matches"]]
    return []

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>AI-Powered Legal HelpDesk</h1>", unsafe_allow_html=True)

pdf_source = st.radio("Select PDF Source", ("Upload from PC", "Choose from Document Storage"))
if pdf_source == "Upload from PC":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        temp_pdf_path = f"temp_{uploaded_file.name}"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.read())
        
        articles = process_pdf(temp_pdf_path, uploaded_file.name)
        store_vectors(articles)
        st.success("PDF uploaded and processed successfully!")

st.subheader("Choose Input Language")
input_language = st.selectbox("Select Input Language", ("English", "Arabic"))

st.subheader("Choose Response Language")
response_language = st.selectbox("Select Response Language", ("English", "Arabic"))

query = st.text_input("Ask a question:")
if st.button("Get Answer"):
    if query and uploaded_file:
        responses = query_vectors(query, uploaded_file.name)
        for res in responses:
            st.write(f"**Chapter {res['chapter']} - Article {res['article_number']}:** {res['text']}")
    else:
        st.warning("Please upload a PDF and enter a query.")
