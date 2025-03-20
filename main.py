import streamlit as st
from PyPDF2 import PdfReader
import pinecone
from sentence_transformers import SentenceTransformer
import os

# 🌟 Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "helpdesk"
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Load SentenceTransformer model for embeddings
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 🌟 Streamlit UI
st.title("📜 Legal HelpDesk for Saudi Arabia")

# PDF Source Selection
st.subheader("Select PDF Source")
pdf_option = st.radio("Choose a source:", ["Upload from PC", "Choose from the Document Storage"])

# File upload or storage selection
uploaded_pdf = None
preloaded_pdfs = {"Law of the Council of Ministers.pdf": "data/law_council_ministers.pdf"}
selected_pdf = None
pdf_text = ""

if pdf_option == "Upload from PC":
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

if pdf_option == "Choose from the Document Storage":
    selected_pdf = st.selectbox("Select a PDF", list(preloaded_pdfs.keys()))

# Extract and store text in Pinecone
def extract_and_store_pdf(pdf_path, pdf_name):
    """Extracts text from the PDF, splits into articles, and stores them in Pinecone."""
    pdf_reader = PdfReader(pdf_path)
    text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    
    # Split PDF text into articles
    articles = text.split("Article ")
    articles = ["Article " + art.strip() for art in articles if art.strip()]
    
    # Generate embeddings and store in Pinecone
    vectors = []
    for i, article in enumerate(articles):
        embedding = embedding_model.encode(article).tolist()
        vectors.append((f"{pdf_name}_article_{i+1}", embedding, {"text": article, "pdf_name": pdf_name}))

    index.upsert(vectors)
    return "✅ Successfully stored in Pinecone!"

if uploaded_pdf:
    st.write(extract_and_store_pdf(uploaded_pdf, uploaded_pdf.name))
elif selected_pdf:
    st.write(extract_and_store_pdf(preloaded_pdfs[selected_pdf], selected_pdf))

# Language Selection
st.subheader("Choose Input Language")
input_lang = st.radio("Input Language:", ["English", "Arabic"])

st.subheader("Choose Response Language")
response_lang = st.radio("Response Language:", ["English", "Arabic"])

# User query input
st.subheader("Ask a question (in English or Arabic):")
query = st.text_input("Enter your legal query:")

# Search Pinecone for relevant articles
if query:
    query_embedding = embedding_model.encode(query).tolist()
    search_results = index.query(query_embedding, top_k=1, include_metadata=True)

    if search_results["matches"]:
        best_match = search_results["matches"][0]
        st.subheader("📖 Relevant Legal Article:")
        st.write(best_match["metadata"]["text"])
    else:
        st.warning("No relevant articles found.")
