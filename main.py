import streamlit as st
import pinecone
import os
import PyPDF2
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import torch

# Load environment variables
load_dotenv()

# Pinecone setup
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index_name = "helpdesk"
index = pc.Index(index_name)

# Load Hugging Face Embedding Model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def get_embedding(text):
    """Generate embeddings using Hugging Face model."""
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        output = model(**tokens).last_hidden_state.mean(dim=1)
    return output.numpy().tolist()[0]

def process_pdf(pdf_path):
    """Extract articles and their metadata from the PDF."""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

    articles = []
    current_chapter = "Unknown Chapter"
    
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("Chapter"):  
            current_chapter = line  

        if line.startswith("Article"):  
            parts = line.split(" ", 2)
            if len(parts) > 2:
                article_number = parts[1]
                article_text = parts[2]
                articles.append({
                    "article_number": article_number,
                    "chapter_number": current_chapter,
                    "text": article_text
                })

    return articles

def store_articles(articles, pdf_name):
    """Store extracted articles in Pinecone."""
    vectors = []
    for i, article in enumerate(articles):
        vector = get_embedding(article["text"])
        metadata = {
            "article_number": article["article_number"],
            "chapter_number": article["chapter_number"],
            "pdf_name": pdf_name,
            "text": article["text"],
            "type": "article"
        }
        vectors.append((f"{pdf_name}-article-{i}", vector, metadata))
    
    index.upsert(vectors)

def query_pinecone(query):
    """Retrieve top-matching articles from Pinecone."""
    query_vector = get_embedding(query)
    results = index.query(vector=query_vector, top_k=5, include_metadata=True)

    if results["matches"]:
        return results["matches"]
    return []

# Streamlit UI
st.set_page_config(page_title="AI-Powered Legal HelpDesk 🇸🇦", layout="wide")

st.markdown("<h1 style='text-align: center;'>🤖 AI-Powered Legal HelpDesk for Saudi Arabia 🇸🇦</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Ask legal questions and retrieve relevant laws</h3>", unsafe_allow_html=True)

# Sidebar for language selection
st.sidebar.header("🔤 Language Settings")
input_lang = st.sidebar.radio("Choose Input Language", ["English", "Arabic"])
response_lang = st.sidebar.radio("Choose Response Language", ["English", "Arabic"])

# PDF Upload
st.subheader("📂 Select PDF Source")
option = st.radio("Upload from:", ["Upload from PC", "Choose from the Document Storage"])

if option == "Upload from PC":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if option == "Choose from the Document Storage":
    # This can be linked to an actual document storage system
    st.warning("📁 Document storage integration is not yet implemented.")

# Process PDF
if uploaded_file:
    pdf_name = uploaded_file.name
    temp_pdf_path = f"temp_{pdf_name}"
    
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    articles = process_pdf(temp_pdf_path)
    store_articles(articles, pdf_name)
    st.success("📑 PDF processed and articles stored successfully!")

# User Query
st.subheader("💬 Ask a Legal Question")
query = st.text_input("Enter your question (in English or Arabic):")
if st.button("Search 🔎"):
    if query:
        results = query_pinecone(query)
        if results:
            for i, match in enumerate(results):
                st.write(f"### {i+1}. **Article {match['metadata']['article_number']}**")
                st.write(f"📖 **Chapter:** {match['metadata']['chapter_number']}")
                st.write(f"📄 **Text:** {match['metadata']['text']}")
                st.write("---")
        else:
            st.warning("⚠️ No relevant articles found.")
    else:
        st.warning("⚠️ Please enter a query!")

