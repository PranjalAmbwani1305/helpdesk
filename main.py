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

# Function to extract Chapters & Articles properly
def process_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

    sections = []
    current_chapter = None
    current_article = None
    current_content = []

    chapter_pattern = r'^(Chapter \w+):'
    article_pattern = r'^(Article \d+|Article [A-Za-z]+):'

    paragraphs = text.split('\n')

    for para in paragraphs:
        para = para.strip()
        
        # Check if it's a chapter
        if re.match(chapter_pattern, para):
            if current_article:
                sections.append({'chapter': current_chapter, 'article': current_article, 'content': ' '.join(current_content)})
                current_article = None
                current_content = []

            current_chapter = para
            continue

        # Check if it's an article
        if re.match(article_pattern, para):
            if current_article:
                sections.append({'chapter': current_chapter, 'article': current_article, 'content': ' '.join(current_content)})

            current_article = para
            current_content = []
        else:
            current_content.append(para)

    # Store last section
    if current_article:
        sections.append({'chapter': current_chapter, 'article': current_article, 'content': ' '.join(current_content)})

    return sections

# Function to store data in Pinecone
def store_vectors(sections, pdf_name):
    for i, section in enumerate(sections):
        chapter = section['chapter']
        article = section['article']
        content = section['content']

        chapter_vector = model.encode(chapter).tolist() if chapter else None
        article_vector = model.encode(article).tolist()
        content_vector = model.encode(content).tolist()

        # Store in Pinecone
        upserts = [
            (f"{pdf_name}-ch-{i}", chapter_vector, {"pdf_name": pdf_name, "text": chapter, "type": "chapter"}) if chapter else None,
            (f"{pdf_name}-art-{i}", article_vector, {"pdf_name": pdf_name, "text": article, "type": "article"}),
            (f"{pdf_name}-content-{i}", content_vector, {"pdf_name": pdf_name, "text": content, "type": "content"})
        ]
        
        upserts = [x for x in upserts if x is not None]
        index.upsert(upserts)

# Fetch previously uploaded PDFs from Pinecone
def get_uploaded_pdfs():
    try:
        result = index.describe_index_stats()
        if "namespaces" in result:
            return list(result["namespaces"].keys())
    except Exception:
        return []
    return []

# Query function to retrieve relevant data
def query_vectors(query, selected_pdf):
    try:
        vector = model.encode(query).tolist()
        results = index.query(vector=vector, top_k=5, include_metadata=True, filter={"pdf_name": {"$eq": selected_pdf}})
        
        if results["matches"]:
            matched_texts = [match["metadata"]["text"] for match in results["matches"]]
            return "\n\n".join(matched_texts)
        else:
            return "No relevant information found in the selected document."
    except Exception as e:
        return f"Error during query: {e}"

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>AI-Powered Legal HelpDesk</h1>", unsafe_allow_html=True)

# Select PDF Source: Upload or Choose from Previous
pdf_source = st.radio("Select PDF Source", ("Upload from PC", "Choose from Document Storage"))

if pdf_source == "Upload from PC":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        temp_pdf_path = f"temp_{uploaded_file.name}"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        sections = process_pdf(temp_pdf_path)
        store_vectors(sections, uploaded_file.name)
        st.success("PDF uploaded and processed!")
        selected_pdf = uploaded_file.name  # Set selected PDF
else:
    stored_pdfs = get_uploaded_pdfs()
    if stored_pdfs:
        selected_pdf = st.selectbox("Select a previously uploaded PDF:", stored_pdfs)
    else:
        st.warning("No stored PDFs found. Please upload one.")

# Language Selection
input_language = st.selectbox("Choose Input Language", ("English", "Arabic"))
response_language = st.selectbox("Choose Response Language", ("English", "Arabic"))

# Query Input and Processing
query = st.text_input("Ask a legal question:")

if st.button("Get Answer"):
    if query and selected_pdf:
        response = query_vectors(query, selected_pdf)
        st.write(f"**Answer:** {response}")
    else:
        st.warning("Please upload/select a PDF and enter a query.")
