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

# Function to process PDF and extract structured articles
def process_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])

    sections = []
    current_chapter = None
    current_article_number = None
    current_article_title = None
    current_content = []

    chapter_pattern = r'^(Chapter \w+):\s*(.*)$'  # Match Chapter One: Title
    article_pattern = r'^(Article \d+):\s*(.*)$'  # Match Article 1: Title

    paragraphs = text.split("\n")

    for para in paragraphs:
        para = para.strip()

        # Check if paragraph is a new chapter
        chapter_match = re.match(chapter_pattern, para)
        if chapter_match:
            if current_article_number:
                sections.append({
                    "chapter": current_chapter,
                    "article_number": current_article_number,
                    "article_title": current_article_title,
                    "content": " ".join(current_content)
                })
            current_chapter = chapter_match.group(1)
            current_article_number = None
            current_content = []
            continue

        # Check if paragraph is a new article
        article_match = re.match(article_pattern, para)
        if article_match:
            if current_article_number:
                sections.append({
                    "chapter": current_chapter,
                    "article_number": current_article_number,
                    "article_title": current_article_title,
                    "content": " ".join(current_content)
                })
            current_article_number = article_match.group(1)
            current_article_title = article_match.group(2)
            current_content = []
            continue

        # Otherwise, append paragraph to current article content
        if current_article_number:
            current_content.append(para)

    # Store last article
    if current_article_number:
        sections.append({
            "chapter": current_chapter,
            "article_number": current_article_number,
            "article_title": current_article_title,
            "content": " ".join(current_content)
        })

    return sections

# Function to store extracted articles in Pinecone
def store_vectors(sections, pdf_name):
    for section in sections:
        chapter = section["chapter"]
        article_number = section["article_number"]
        article_title = section["article_title"]
        content = section["content"]

        vector = model.encode(article_number + " " + article_title + " " + content).tolist()

        index.upsert([
            (f"{pdf_name}-chapter-{chapter}-article-{article_number}", vector,
             {"pdf_name": pdf_name, "chapter": chapter, "article_number": article_number, 
              "article_title": article_title, "content": content})
        ])

# Function to query vectors from Pinecone
def query_vectors(query, selected_pdf):
    try:
        vector = model.encode(query).tolist()
        results = index.query(vector=vector, top_k=5, include_metadata=True, filter={"pdf_name": {"$eq": selected_pdf}})

        if results["matches"]:
            response_text = ""
            for match in results["matches"]:
                metadata = match["metadata"]
                chapter = metadata.get("chapter", "Unknown Chapter")
                article_number = metadata.get("article_number", "Unknown Article")
                article_title = metadata.get("article_title", "")
                content = metadata.get("content", "")

                response_text += f"**{chapter} - {article_number}: {article_title}**\n{content}\n\n"

            return response_text if response_text else "No relevant information found in the selected document."
        else:
            return "No relevant information found in the selected document."

    except Exception as e:
        return f"Error during query: {e}"

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>AI-Powered Legal HelpDesk</h1>", unsafe_allow_html=True)

# Sidebar for uploaded PDFs
st.sidebar.header("Uploaded PDFs")
pdf_source = st.radio("Select PDF Source", ("Upload from PC", "Choose from Document Storage"))

if "uploaded_pdfs" not in st.session_state:
    st.session_state.uploaded_pdfs = []

if pdf_source == "Upload from PC":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        temp_pdf_path = f"temp_{uploaded_file.name}"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        sections = process_pdf(temp_pdf_path)
        store_vectors(sections, uploaded_file.name)
        st.session_state.uploaded_pdfs.append(uploaded_file.name)
        st.success("PDF uploaded and processed!")

# Sidebar dropdown to select uploaded PDF
if st.session_state.uploaded_pdfs:
    selected_pdf = st.sidebar.selectbox("Select a PDF", st.session_state.uploaded_pdfs)
else:
    selected_pdf = ""

# Language Selection
input_language = st.selectbox("Choose Input Language", ("English", "Arabic"))
response_language = st.selectbox("Choose Response Language", ("English", "Arabic"))

# Query Input
query = st.text_input("Ask a legal question:")

if st.button("Get Answer"):
    if query:
        response = query_vectors(query, selected_pdf)
        st.write(f"**Answer:** {response}")
    else:
        st.warning("Please upload a PDF and enter a query.")
