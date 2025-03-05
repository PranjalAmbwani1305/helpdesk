import streamlit as st
import pinecone
import PyPDF2
import os
import re
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
index_name = "helpdesk"

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(index_name)

# Initialize Sentence Transformer
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize session state for uploaded PDFs
if "uploaded_pdfs" not in st.session_state:
    st.session_state["uploaded_pdfs"] = {}

# Function to extract and structure PDF content into Chapters and Articles
def process_pdf(pdf_path, pdf_name):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

    sections = []
    current_chapter = None
    current_article = None
    chapter_content = []
    article_content = []

    chapter_pattern = r'^(Chapter \d+|Chapter [A-Za-z]+):\s*(.*)$'
    article_pattern = r'^(Article \d+|Article [A-Za-z]+):\s*(.*)$'

    paragraphs = text.split("\n")

    for para in paragraphs:
        para = para.strip()
        
        chapter_match = re.match(chapter_pattern, para)
        article_match = re.match(article_pattern, para)

        if chapter_match:
            if current_chapter:
                sections.append({
                    'title': f"{pdf_name} - {current_chapter}",
                    'content': ' '.join(chapter_content)
                })

            current_chapter = f"{chapter_match.group(1)}: {chapter_match.group(2)}"
            chapter_content = []
            current_article = None  # Reset article when a new chapter starts

        elif article_match:
            if current_article:
                sections.append({
                    'title': f"{pdf_name} - {current_article}",
                    'content': ' '.join(article_content)
                })

            current_article = f"{article_match.group(1)}: {article_match.group(2)}"
            article_content = []

        else:
            if current_article:
                article_content.append(para)
            elif current_chapter:
                chapter_content.append(para)

    if current_article:
        sections.append({'title': f"{pdf_name} - {current_article}", 'content': ' '.join(article_content)})
    elif current_chapter:
        sections.append({'title': f"{pdf_name} - {current_chapter}", 'content': ' '.join(chapter_content)})

    return sections

# Function to store vectors in Pinecone
def store_vectors(sections, pdf_name):
    for i, section in enumerate(sections):
        title = section["title"]
        content = section["content"]
        
        title_vector = model.encode(title).tolist()
        content_vector = model.encode(content).tolist()
        
        index.upsert([
            (f"{pdf_name}-section-{i}-title", title_vector, {
                "pdf_name": pdf_name, 
                "section": title,
                "text": title, 
                "type": "title"
            }),
            (f"{pdf_name}-section-{i}-content", content_vector, {
                "pdf_name": pdf_name, 
                "section": title,
                "text": content, 
                "type": "content"
            })
        ])

# Function to query vectors from Pinecone
def query_vectors(query, selected_pdf):
    try:
        vector = model.encode(query).tolist()
        results = index.query(vector=vector, top_k=5, include_metadata=True, filter={"pdf_name": {"$eq": selected_pdf}})
        
        if results["matches"]:
            matched_texts = [f"**{match['metadata']['section']}**\n\n{match['metadata']['text']}" for match in results["matches"]]
            return "\n\n".join(matched_texts)
        else:
            return "No relevant information found in the selected document."
    except Exception as e:
        return f"Error during query: {e}"

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>📜 AI-Powered Legal HelpDesk</h1>", unsafe_allow_html=True)

# Sidebar for PDF selection
st.sidebar.header("📂 Uploaded PDFs")
if st.session_state["uploaded_pdfs"]:
    selected_pdf = st.sidebar.radio("Select a PDF", list(st.session_state["uploaded_pdfs"].keys()))
else:
    selected_pdf = None
    st.sidebar.warning("No PDFs uploaded.")

# PDF Upload Section
pdf_source = st.radio("Select PDF Source", ("Upload from PC", "Choose from Document Storage"))

if pdf_source == "Upload from PC":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    
    if uploaded_file:
        if uploaded_file.name not in st.session_state["uploaded_pdfs"]:
            temp_pdf_path = f"temp_{uploaded_file.name}"
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.read())
            
            sections = process_pdf(temp_pdf_path, uploaded_file.name)
            store_vectors(sections, uploaded_file.name)
            
            st.session_state["uploaded_pdfs"][uploaded_file.name] = True  # Mark as uploaded
            st.sidebar.success(f"✅ {uploaded_file.name} uploaded!")
            st.experimental_rerun()  # Refresh UI to show the uploaded PDF

# Query Section
query = st.text_input("🔍 Ask a legal question:")

if st.button("Get Answer"):
    if not selected_pdf:
        st.warning("⚠️ Please select or upload a PDF first.")
    elif query:
        response = query_vectors(query, selected_pdf)
        st.write(f"**Answer:**\n\n{response}")
    else:
        st.warning("⚠️ Please enter a question.")

# Display uploaded PDFs in sidebar
st.sidebar.markdown("### 🗂 Uploaded Documents")
for pdf in st.session_state["uploaded_pdfs"]:
    st.sidebar.write(f"📄 {pdf}")
