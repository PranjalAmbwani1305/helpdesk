import streamlit as st
import pinecone
import PyPDF2
import os
import json
import re
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables for Pinecone API key
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "helpdesk"

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(INDEX_NAME)

# Initialize sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# File to store uploaded PDFs persistently
PDF_STORAGE_FILE = "uploaded_pdfs.json"

def load_uploaded_pdfs():
    if os.path.exists(PDF_STORAGE_FILE):
        with open(PDF_STORAGE_FILE, "r") as f:
            return json.load(f)
    return []

def save_uploaded_pdfs(pdf_list):
    with open(PDF_STORAGE_FILE, "w") as f:
        json.dump(pdf_list, f)

uploaded_pdfs = load_uploaded_pdfs()

# Function to process PDF, extract chapters and articles
def process_pdf(pdf_path, pdf_name):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
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
                sections.append({
                    "chapter": current_chapter,
                    "article": current_article,
                    "content": " ".join(current_content)
                })
            current_chapter = para
            current_article = None
            current_content = []
        
        elif re.match(article_pattern, para):
            if current_article:
                sections.append({
                    "chapter": current_chapter,
                    "article": current_article,
                    "content": " ".join(current_content)
                })
            current_article = para
            current_content = []
        
        else:
            current_content.append(para)
    
    if current_article:
        sections.append({
            "chapter": current_chapter,
            "article": current_article,
            "content": " ".join(current_content)
        })
    
    return sections

# Function to store vectors in Pinecone
def store_vectors(sections, pdf_name):
    for i, section in enumerate(sections):
        title = f"{section['chapter']} - {section['article']}"
        content = section['content']
        
        title_vector = model.encode(title).tolist()
        content_vector = model.encode(content).tolist()
        
        # Debugging logs
        print(f"Upserting: {pdf_name}-section-{i}")
        
        index.upsert([
            (f"{pdf_name}-section-{i}-title", title_vector, {
                "pdf_name": pdf_name,
                "chapter": section['chapter'],
                "article": section['article'],
                "text": title,
                "type": "title"
            }),
            (f"{pdf_name}-section-{i}-content", content_vector, {
                "pdf_name": pdf_name,
                "chapter": section['chapter'],
                "article": section['article'],
                "text": content,
                "type": "content"
            })
        ])

# Function to query vectors from Pinecone
def query_vectors(query, selected_pdf):
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
        store_vectors(sections, uploaded_file.name)
        
        if uploaded_file.name not in uploaded_pdfs:
            uploaded_pdfs.append(uploaded_file.name)
            save_uploaded_pdfs(uploaded_pdfs)
        
        st.success("PDF uploaded and processed!")
else:
    st.info("Document Storage feature is currently unavailable.")

# Language Selection
input_language = st.selectbox("Choose Input Language", ("English", "Arabic"))
response_language = st.selectbox("Choose Response Language", ("English", "Arabic"))

# Query Input and Processing
query = st.text_input("Ask a legal question:")
if st.button("Get Answer"):
    if query:
        response = query_vectors(query, selected_pdf if selected_pdf != "No PDFs uploaded" else "")
        st.write(f"**Answer:** {response}")
    else:
        st.warning("Please upload a PDF and enter a query.")

