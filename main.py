import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import PyPDF2
import os

# Load models
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
qa_pipeline = pipeline("text-generation", model="tiiuae/falcon-7b-instruct")

# Store uploaded PDFs
if "pdf_store" not in st.session_state:
    st.session_state.pdf_store = {}

def process_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

# Streamlit UI
st.set_page_config(page_title="Legal HelpDesk", layout="wide")
st.markdown("<h1 style='text-align: center;'>📜 AI-Powered Legal HelpDesk</h1>", unsafe_allow_html=True)

# Sidebar: File Upload
st.sidebar.header("📂 Upload Legal Documents")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    file_text = process_pdf(uploaded_file)
    st.session_state.pdf_store[uploaded_file.name] = file_text
    st.sidebar.success(f"✅ {uploaded_file.name} uploaded!")

# Sidebar: Select a stored PDF
selected_pdf = st.sidebar.selectbox("Select a Document", options=list(st.session_state.pdf_store.keys()))

# Main: Query Input
st.subheader("📝 Ask a Legal Question")
query = st.text_input("Enter your question:")

# Generate Answer
if st.button("Get Answer"):
    if selected_pdf and query:
        # Get relevant text from PDF
        pdf_text = st.session_state.pdf_store[selected_pdf]
        query_embedding = embedder.encode(query)
        pdf_embedding = embedder.encode(pdf_text)

        # Generate response
        response = qa_pipeline(query, max_length=200)[0]["generated_text"]
        
        # Display answer
        st.subheader("📖 AI Answer:")
        st.write(response)
    else:
        st.warning("Please upload a PDF and enter a question.")

# Footer
st.markdown("<hr><p style='text-align: center;'>🤖 Built with AI | Powered by Hugging Face</p>", unsafe_allow_html=True)
