import streamlit as st
import pinecone
import PyPDF2
import os
import time
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import re

# Load environment variables for Pinecone API key
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "helpdesk"

if not PINECONE_API_KEY:
    st.error("Pinecone API key is missing. Please check your .env file.")
    st.stop()

# Initialize Pinecone
from pinecone import Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if index exists, if not, create it
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(name=INDEX_NAME, dimension=384, metric="cosine")

index = pc.Index(INDEX_NAME)

# Initialize sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# File to store uploaded PDFs persistently
PDF_STORAGE_FILE = "uploaded_pdfs.txt"

def load_uploaded_pdfs():
    """Load list of previously uploaded PDFs from file."""
    if os.path.exists(PDF_STORAGE_FILE):
        with open(PDF_STORAGE_FILE, "r") as f:
            return f.read().splitlines()
    return []

def save_uploaded_pdfs(pdf_list):
    """Save the list of uploaded PDFs to file."""
    with open(PDF_STORAGE_FILE, "w") as f:
        f.write("\n".join(pdf_list))

uploaded_pdfs = load_uploaded_pdfs()

def process_pdf(pdf_path, pdf_name):
    """Extract chapters and articles from a PDF file."""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
    sections = []
    current_chapter = None
    current_article = None
    current_content = []

    chapter_pattern = r'^(Chapter \w+: .*)$'
    article_pattern = r'^(Article \d+: .*)$'

    paragraphs = text.split('\n')

    for para in paragraphs:
        para = para.strip()
        
        if re.match(chapter_pattern, para):
            if current_article:
                sections.append({"chapter": current_chapter, "article": current_article, "content": " ".join(current_content)})
            current_chapter = para
            current_article = None
            current_content = []
        elif re.match(article_pattern, para):
            if current_article:
                sections.append({"chapter": current_chapter, "article": current_article, "content": " ".join(current_content)})
            current_article = para
            current_content = []
        else:
            current_content.append(para)
    
    if current_article:
        sections.append({"chapter": current_chapter, "article": current_article, "content": " ".join(current_content)})
    
    return sections

def store_vectors(sections, pdf_name):
    """Store extracted sections as embeddings in Pinecone."""
    vectors = []
    for i, section in enumerate(sections):
        title = f"{section['chapter']} - {section['article']}"
        content = section['content']
        
        title_vector = model.encode(title).tolist()
        content_vector = model.encode(content).tolist()
        
        vectors.append((f"{pdf_name}-section-{i}-title", title_vector, {"pdf_name": pdf_name, "chapter": section['chapter'], "article": section['article'], "text": title, "type": "title"}))
        vectors.append((f"{pdf_name}-section-{i}-content", content_vector, {"pdf_name": pdf_name, "chapter": section['chapter'], "article": section['article'], "text": content, "type": "content"}))

    # Store in Pinecone in smaller chunks
    BATCH_SIZE = 100
    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i:i + BATCH_SIZE]
        index.upsert(batch)
        time.sleep(1)  # Prevent rate limits

    print(f"Stored {len(vectors)} vectors in Pinecone.")

def query_vectors(query, selected_pdf):
    """Retrieve the most relevant results from Pinecone based on the query."""
    vector = model.encode(query).tolist()
    results = index.query(vector=vector, top_k=5, include_metadata=True, filter={"pdf_name": {"$eq": selected_pdf}})
    
    if results["matches"]:
        response_text = ""
        for match in results["matches"]:
            response_text += f"**{match['metadata']['chapter']} - {match['metadata']['article']}:**\n{match['metadata']['text']}\n\n"
        return response_text
    return "No relevant information found in the selected document."

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>AI-Powered Legal HelpDesk</h1>", unsafe_allow_html=True)

# Sidebar to show uploaded PDFs
st.sidebar.header("Uploaded PDFs")
selected_pdf = st.sidebar.selectbox("Select a PDF", uploaded_pdfs if uploaded_pdfs else ["No PDFs uploaded"])

# PDF Upload Section
pdf_source = st.radio("Select PDF Source", ("Upload from PC", "Choose from Document Storage"))
if pdf_source == "Upload from PC":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        temp_pdf_path = f"temp_{uploaded_file.name}"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.read())
        
        sections = process_pdf(temp_pdf_path, uploaded_file.name)
        
        if sections:
            store_vectors(sections, uploaded_file.name)
            if uploaded_file.name not in uploaded_pdfs:
                uploaded_pdfs.append(uploaded_file.name)
                save_uploaded_pdfs(uploaded_pdfs)
            st.success("PDF uploaded and processed!")
        else:
            st.error("No valid sections found in the PDF.")

else:
    st.info("Document Storage feature is currently unavailable.")

# Language Selection
input_language = st.selectbox("Choose Input Language", ("English", "Arabic"))
response_language = st.selectbox("Choose Response Language", ("English", "Arabic"))

# Query Input and Processing
query = st.text_input("Ask a legal question:")
if st.button("Get Answer"):
    if query:
        if selected_pdf and selected_pdf != "No PDFs uploaded":
            response = query_vectors(query, selected_pdf)
            st.write(f"**Answer:** {response}")
        else:
            st.warning("Please upload a PDF first.")
    else:
        st.warning("Please enter a query.")
