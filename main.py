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
index = pc.Index(index_name)

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to extract and structure content from PDFs
def process_pdf(pdf_path, chunk_size=500):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])

    sections = []
    current_chapter = None
    current_article = None
    current_content = []

    chapter_pattern = r'^(Chapter \d+|Chapter [A-Za-z]+):.*$'
    article_pattern = r'^(Article \d+|Article [A-Za-z]+):.*$'

    paragraphs = text.split("\n")

    for para in paragraphs:
        para = para.strip()

        if re.match(chapter_pattern, para):
            current_chapter = para  # Set new chapter
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
        chapter = section['chapter'] if section['chapter'] else "Unknown Chapter"
        article = section['article'] if section['article'] else "Unknown Article"
        content = section['content']

        article_vector = model.encode(article).tolist()
        content_vector = model.encode(content).tolist()

        index.upsert([
            (f"{pdf_name}-section-{i}-article", article_vector, {
                "pdf_name": pdf_name, "chapter": chapter, "article": article, "text": article, "type": "article"
            }),
            (f"{pdf_name}-section-{i}-content", content_vector, {
                "pdf_name": pdf_name, "chapter": chapter, "article": article, "text": content, "type": "content"
            })
        ])

# Function to query Pinecone
def query_vectors(query, selected_pdf):
    vector = model.encode(query).tolist()
    
    # Check if query is asking for a specific article
    article_match = re.match(r'^(Explain|What is) (Article \d+|Article [A-Za-z]+)', query.strip(), re.IGNORECASE)
    
    if article_match:
        article_name = article_match.group(2)
        results = index.query(vector=vector, top_k=5, include_metadata=True, filter={
            "pdf_name": {"$eq": selected_pdf}, 
            "article": {"$eq": article_name}
        })
    else:
        results = index.query(vector=vector, top_k=5, include_metadata=True, filter={
            "pdf_name": {"$eq": selected_pdf}
        })

    matched_articles = {}

    if results["matches"]:
        for match in results["matches"]:
            pdf_name = match["metadata"].get("pdf_name", "Unknown PDF")
            chapter = match["metadata"].get("chapter", "Unknown Chapter")
            article = match["metadata"].get("article", "Unknown Article")
            entry_type = match["metadata"].get("type", "")

            if article not in matched_articles:
                matched_articles[article] = {
                    "pdf_name": pdf_name,
                    "chapter": chapter,
                    "article": article,
                    "content": ""
                }

            if entry_type == "content":
                matched_articles[article]["content"] = match["metadata"].get("text", "")

        response_text = "\n\n".join([
            f"📄 **PDF:** {entry['pdf_name']}\n📖 **Chapter:** {entry['chapter']}\n📝 **{entry['article']}**\n{entry['content']}"
            for entry in matched_articles.values()
        ])
        return response_text
    else:
        return "No relevant information found in the selected document."

# Streamlit UI
st.sidebar.title("Uploaded PDFs")

pdf_source = st.radio("Select PDF Source", ("Upload from PC", "Choose from Document Storage"))

uploaded_file = None
selected_pdf = None

if pdf_source == "Upload from PC":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        selected_pdf = uploaded_file.name
        temp_pdf_path = f"temp_{uploaded_file.name}"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        sections = process_pdf(temp_pdf_path)
        store_vectors(sections, uploaded_file.name)
        st.sidebar.write(f"✅ {uploaded_file.name}")
        st.success("PDF uploaded and processed!")

# Language Selection
input_language = st.selectbox("Choose Input Language", ("English", "Arabic"))
response_language = st.selectbox("Choose Response Language", ("English", "Arabic"))

# Query Input and Processing
query = st.text_input("Ask a legal question:")

if st.button("Get Answer"):
    if query:
        response = query_vectors(query, selected_pdf if uploaded_file else "")
        st.write(f"**Answer:** {response}")
    else:
        st.warning("Please upload a PDF and enter a query.")
