import os
import pinecone
import streamlit as st
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "helpdesk"

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Load SentenceTransformer model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Function to get stored PDFs from Pinecone
def get_stored_pdfs():
    """Fetch all stored PDF names from Pinecone."""
    results = index.query(vector=[0] * 384, top_k=1000, include_metadata=True)
    pdf_names = {res["metadata"]["pdf_name"] for res in results["matches"] if "pdf_name" in res["metadata"]}
    return list(pdf_names)

# Function to fetch articles from a selected PDF
def fetch_article_from_pdf(selected_pdf, query):
    """Retrieve only the article that matches the user's query from the selected PDF."""
    query_vector = embedding_model.encode(query).tolist()
    response = index.query(vector=query_vector, top_k=5, include_metadata=True, filter={"pdf_name": selected_pdf})
    
    if response["matches"]:
        return response["matches"][0]["metadata"].get("article_text", "No relevant article found.")
    return "No relevant article found."

# Streamlit UI
st.title("📖 Legal HelpDesk for Saudi Arabia")

# Select PDF Source
pdf_source = st.radio("Select PDF Source", ["Upload from PC", "Choose from the Document Storage"])

selected_pdf = None  # Initialize selected_pdf

if pdf_source == "Choose from the Document Storage":
    stored_pdfs = get_stored_pdfs()
    
    if stored_pdfs:
        selected_pdf = st.selectbox("Select a PDF", stored_pdfs, index=0)
        st.success(f"📌 Selected: {selected_pdf}")
    else:
        st.warning("No PDFs found in storage.")
else:
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        selected_pdf = uploaded_file.name
        st.success(f"📌 Selected: {selected_pdf}")

# Choose Input and Response Language
col1, col2 = st.columns(2)
with col1:
    input_language = st.radio("Choose Input Language", ["English", "Arabic"], key="input_lang")
with col2:
    response_language = st.radio("Choose Response Language", ["English", "Arabic"], key="response_lang")

# Ask a Question
user_query = st.text_input("Ask a question (in English or Arabic):")

# Fetch and Display Answer
if user_query and selected_pdf:
    result = fetch_article_from_pdf(selected_pdf, user_query)
    st.markdown(f"### 📜 Answer: \n{result}")
elif user_query:
    st.error("⚠️ Please select a PDF before asking a question.")
