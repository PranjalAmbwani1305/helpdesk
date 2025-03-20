import os
import pinecone
import streamlit as st
from sentence_transformers import SentenceTransformer

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "helpdesk"

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Load SentenceTransformer model for embeddings
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Function to get stored PDFs from Pinecone
def get_stored_pdfs():
    """Fetch unique PDF names from Pinecone metadata."""
    results = index.query(vector=[0] * 384, top_k=1000, include_metadata=True)
    pdf_names = list({res["metadata"]["pdf_name"] for res in results["matches"] if "pdf_name" in res["metadata"]})
    return pdf_names

# Function to retrieve an article from the selected PDF
def fetch_article_from_pdf(selected_pdf, query):
    """Retrieve the most relevant article from the selected PDF in Pinecone."""
    
    query_vector = embedding_model.encode(query).tolist()

    response = index.query(
        vector=query_vector, 
        top_k=5, 
        include_metadata=True, 
        filter={"pdf_name": {"$eq": selected_pdf}}
    )

    if response and response["matches"]:
        best_match = response["matches"][0]["metadata"].get("article_text", "No relevant article found.")
        return best_match
    return "No relevant article found."

# Streamlit UI
st.title("📖 Legal HelpDesk for Saudi Arabia")

# Choose from stored PDFs
stored_pdfs = get_stored_pdfs()

if stored_pdfs:
    selected_pdf = st.selectbox("Select a PDF (from Pinecone)", stored_pdfs, index=0)
    st.success(f"📌 Selected: {selected_pdf}")
else:
    st.warning("No PDFs found in Pinecone storage.")

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
