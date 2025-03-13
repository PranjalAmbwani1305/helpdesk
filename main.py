import os
import pinecone
import pdfplumber
import streamlit as st
from tempfile import NamedTemporaryFile
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")  
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "helpdesk"

index = pc.Index(index_name)

# Load Sentence-Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract articles from a PDF
def extract_articles(pdf_path):
    articles = []
    with pdfplumber.open(pdf_path) as pdf:
        current_chapter = None

        for page in pdf.pages:
            text = page.extract_text()
            if text:
                lines = text.split("\n")
                for line in lines:
                    if "Chapter" in line:
                        current_chapter = line.strip()
                    elif "Article" in line:  # Detect articles
                        articles.append({"title": line.strip(), "chapter": current_chapter, "text": ""})
                    elif articles:  # Append text to last detected article
                        articles[-1]["text"] += line.strip() + " "

    return articles

# Function to store multiple PDFs' articles efficiently
def store_vectors_batch(pdf_articles):
    try:
        upsert_data = []
        
        for pdf_name, articles in pdf_articles.items():
            for i, article in enumerate(articles):
                vector = model.encode(article["text"]).tolist()
                upsert_data.append((
                    f"{pdf_name}-article-{i}",
                    vector,
                    {
                        "pdf_name": pdf_name,
                        "title": article["title"],
                        "chapter": article["chapter"],
                        "text": article["text"],
                        "type": "article"
                    }
                ))

        if upsert_data:
            index.upsert(upsert_data)
            st.success(f"✅ Successfully stored {len(upsert_data)} articles in Pinecone")
    except Exception as e:
        st.error(f"❌ Error storing data: {e}")

# Streamlit UI
st.title("📜 Multi-PDF Legal Document Storage & Search Bot")

# Upload Section (Supports multiple PDFs)
uploaded_files = st.file_uploader("📂 Upload multiple PDF files", accept_multiple_files=True, type="pdf")

if uploaded_files:
    pdf_articles = {}

    for uploaded_file in uploaded_files:
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        articles = extract_articles(tmp_file_path)
        pdf_articles[uploaded_file.name] = articles

    store_vectors_batch(pdf_articles)

# Search UI
query = st.text_input("🔍 Enter a legal query:")

if query:
    query_vector = model.encode(query).tolist()
    results = index.query(vector=query_vector, top_k=5, include_metadata=True)

    st.write("### 📜 Search Results:")
    if "matches" in results:
        for match in results["matches"]:
            st.write(f"📂 **File:** {match['metadata']['pdf_name']}")
            st.write(f"📜 **Chapter:** {match['metadata']['chapter']}")
            st.write(f"🔹 **Title:** {match['metadata']['title']}")
            st.write(f"📖 **Text:** {match['metadata']['text'][:300]}...")  
            st.write(f"📈 **Relevance Score:** {match['score']:.2f}")
            st.write("---")
    else:
        st.warning("⚠️ No relevant results found.")
