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

from pinecone import Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Initialize Pinecone index
index = pc.Index(index_name)

# Initialize sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to process PDF, extract chapters, articles, and content
def process_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])

    sections = []
    current_chapter = None
    current_title = None
    current_content = []

    # Patterns for detecting chapters and articles
    chapter_pattern = r'^(Chapter \w+):.*$'
    article_pattern = r'^(Article \d+|Article [A-Za-z]+):.*$'

    paragraphs = text.split('\n')

    for para in paragraphs:
        para = para.strip()
        
        # Check for Chapter Title
        if re.match(chapter_pattern, para):
            if current_title:  # Store previous section before switching
                sections.append({
                    'chapter': current_chapter,
                    'title': current_title,
                    'content': ' '.join(current_content)
                })
            current_chapter = para
            current_title = None
            current_content = []

        # Check for Article Title
        elif re.match(article_pattern, para):
            if current_title:  # Store previous article before switching
                sections.append({
                    'chapter': current_chapter,
                    'title': current_title,
                    'content': ' '.join(current_content)
                })
            current_title = para
            current_content = []

        # Collect content
        else:
            current_content.append(para)

    # Store last section
    if current_title:
        sections.append({
            'chapter': current_chapter,
            'title': current_title,
            'content': ' '.join(current_content)
        })

    return sections

# Function to store vectors in Pinecone
def store_vectors(sections, pdf_name):
    for i, section in enumerate(sections):
        chapter = section['chapter'] if section['chapter'] else "No Chapter"
        title = section['title']
        content = section['content']
        
        # Encode and store vectors
        title_vector = model.encode(title).tolist()
        content_vector = model.encode(content).tolist()
        
        index.upsert([
            (f"{pdf_name}-section-{i}-title", title_vector, {"pdf_name": pdf_name, "chapter": chapter, "text": title, "type": "title"}),
            (f"{pdf_name}-section-{i}-content", content_vector, {"pdf_name": pdf_name, "chapter": chapter, "text": content, "type": "content"})
        ])

# Function to query vectors from Pinecone
def query_vectors(query, selected_pdf):
    try:
        vector = model.encode(query).tolist()
        results = index.query(vector=vector, top_k=5, include_metadata=True, filter={"pdf_name": {"$eq": selected_pdf}})
        
        if results["matches"]:
            matched_texts = []
            
            for match in results["matches"]:
                chapter = match["metadata"].get("chapter", "No Chapter")
                title = match["metadata"].get("text", "Unknown Article")
                content = match["metadata"].get("content", "No content found")

                matched_texts.append(f"**{chapter}**\n**{title}**\n{content}")

            return "\n\n".join(matched_texts)
        else:
            return "No relevant information found in the selected document."
    except Exception as e:
        return f"Error during query: {e}"

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>AI-Powered Legal HelpDesk</h1>", unsafe_allow_html=True)

# PDF Source Selection: Upload or Choose from Document Storage
pdf_source = st.radio("Select PDF Source", ("Upload from PC", "Choose from Document Storage"))

if "uploaded_pdfs" not in st.session_state:
    st.session_state["uploaded_pdfs"] = {}

if pdf_source == "Upload from PC":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    
    if uploaded_file:
        if uploaded_file.name not in st.session_state["uploaded_pdfs"]:
            temp_pdf_path = f"temp_{uploaded_file.name}"
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.read())
            
            sections = process_pdf(temp_pdf_path)
            store_vectors(sections, uploaded_file.name)
            st.session_state["uploaded_pdfs"][uploaded_file.name] = True  # Mark as uploaded

            st.success("PDF uploaded and processed!")

# Language Selection
input_language = st.selectbox("Choose Input Language", ("English", "Arabic"))
response_language = st.selectbox("Choose Response Language", ("English", "Arabic"))

# Query Input and Processing
query = st.text_input("Ask a legal question:")

if st.button("Get Answer"):
    if query:
        selected_pdf = uploaded_file.name if uploaded_file else ""
        response = query_vectors(query, selected_pdf)
        st.write(f"**Answer:** {response}")
    else:
        st.warning("Please upload a PDF and enter a query.")
