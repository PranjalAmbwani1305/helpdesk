import os
import pinecone
import pdfplumber
import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "helpdesk"
index = pc.Index(index_name)

# Load Sentence-Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract text from PDF
def extract_pdf_content(pdf_path):
    chapters, articles = [], []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                chapters.append(text)  # Store full page text
                articles.append(text)  # Treat each page as an article for simplicity
    return chapters, articles

# Function to store vectors in Pinecone (Ensuring Unique IDs)
def store_vectors(chapters, articles, pdf_name):
    try:
        for i, chapter in enumerate(chapters):
            vector = model.encode(chapter).tolist()
            index.upsert([
                (f"{pdf_name}-chapter-{i}", vector, {"pdf_name": pdf_name, "text": chapter, "type": "chapter"})
            ])

        for i, article in enumerate(articles):
            vector = model.encode(article).tolist()
            index.upsert([
                (f"{pdf_name}-article-{i}", vector, {"pdf_name": pdf_name, "text": article, "type": "article"})
            ])

        print(f"✅ Successfully stored {pdf_name} in Pinecone")

    except Exception as e:
        print(f"❌ Error storing {pdf_name}: {e}")

# Streamlit UI
st.title("Legal Document Storage & Search Bot")

uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type="pdf")

if uploaded_files:
    for uploaded_file in uploaded_files:
        pdf_path = f'/tmp/{uploaded_file.name}'
        with open(pdf_path, 'wb') as f:
            f.write(uploaded_file.read())

        chapters, articles = extract_pdf_content(pdf_path)
        store_vectors(chapters, articles, uploaded_file.name)
        st.success(f"Stored {uploaded_file.name} in Pinecone!")

# Search UI
query = st.text_input("Enter a legal query:")
if query:
    query_vector = model.encode(query).tolist()
    results = index.query(query_vector, top_k=5, include_metadata=True)

    st.write("### Results:")
    for match in results['matches']:
        st.write(f"📜 **{match['metadata']['pdf_name']}**")
        st.write(f"🔍 **Text:** {match['metadata']['text'][:300]}...")  # Show a snippet
        st.write(f"💡 **Score:** {match['score']:.2f}")
