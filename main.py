import streamlit as st
import pinecone
import PyPDF2
import os
import re
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
st.set_page_config(page_title="Legal HelpDesk", layout="wide")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "helpdesk"



pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(INDEX_NAME)

# --- LOAD EMBEDDING MODEL ---
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- UI ELEMENTS ---
st.sidebar.header("📄 Upload Legal Documents")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
st.sidebar.markdown("---")

st.sidebar.header("🔍 Query HelpDesk")
query = st.sidebar.text_input("Enter a legal query", placeholder="e.g., Land acquisition law in India")
query_button = st.sidebar.button("Search")

# --- MAIN LOG DISPLAY ---
st.title("📚 AI-Powered Legal HelpDesk")
log_area = st.empty()

# --- FUNCTIONS ---
def extract_articles_from_pdf(pdf_file, pdf_name):
    """Extracts articles from a PDF and stores them with the PDF name."""
    text = ""
    reader = PyPDF2.PdfReader(pdf_file)
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    # Split text by legal article numbering (customize regex for your document structure)
    articles = re.split(r"(Article \d+:)", text)
    
    # Combine article numbers with their content
    structured_articles = []
    for i in range(1, len(articles), 2):
        article_title = articles[i].strip()
        article_content = articles[i + 1].strip() if i + 1 < len(articles) else ""
        structured_articles.append((pdf_name, article_title, article_content))

    return structured_articles

def store_vectors(pdf_articles):
    """Stores extracted articles as separate vectors in Pinecone."""
    vectors = []
    for pdf_name, article_title, article_content in pdf_articles:
        content_to_store = f"{article_title}\n{article_content}"
        embedding = model.encode(content_to_store).tolist()
        vector_id = f"{pdf_name}_{article_title.replace(' ', '_')}"
        vectors.append((vector_id, embedding, {"pdf": pdf_name, "title": article_title, "content": article_content}))
    
    if vectors:
        index.upsert(vectors)
        log_area.text(f"✅ {len(vectors)} articles stored successfully in Pinecone.")

# --- PDF PROCESSING ---
if uploaded_files:
    all_articles = []
    for pdf_file in uploaded_files:
        pdf_name = pdf_file.name
        articles = extract_articles_from_pdf(pdf_file, pdf_name)
        all_articles.extend(articles)
    
    if all_articles:
        store_vectors(all_articles)

# --- SEARCH FUNCTIONALITY ---
if query_button and query:
    query_embedding = model.encode(query).tolist()
    results = index.query(query_embedding, top_k=5, include_metadata=True)

    st.subheader("🔍 Search Results")
    for match in results["matches"]:
        metadata = match["metadata"]
        st.write(f"📄 **PDF:** {metadata['pdf']}")
        st.write(f"📌 **Article:** {metadata['title']}")
        st.write(f"📝 **Content:** {metadata['content'][:300]}...")  # Show preview
        st.markdown("---")
