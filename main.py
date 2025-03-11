import streamlit as st
import pinecone
import PyPDF2
import os
import re
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
from pinecone import Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "helpdesk"
index = pc.Index(index_name)

# Load Hugging Face Model (Sentence Transformer)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sidebar - Select PDF Source
st.sidebar.title("📂 Select PDF Source")
pdf_source = st.sidebar.radio("Upload or Choose a PDF", ["Upload from PC", "Choose from Document Storage"])

# Function to List Stored PDFs from Pinecone
def get_stored_pdfs():
    try:
        stats = index.describe_index_stats()
        namespaces = stats.get("namespaces", {})
        return list(namespaces.keys())  # Get list of stored PDFs (namespaces)
    except Exception as e:
        st.sidebar.error(f"Error fetching stored PDFs: {e}")
        return []

# Sidebar - Show PDFs Stored in Pinecone
stored_pdfs = get_stored_pdfs()
selected_pdf = None

if pdf_source == "Choose from Document Storage":
    if stored_pdfs:
        selected_pdf = st.sidebar.selectbox("📜 Choose a PDF", stored_pdfs)
    else:
        st.sidebar.warning("No PDFs stored yet. Upload one first.")

# PDF Uploading Logic
if pdf_source == "Upload from PC":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    
    if uploaded_file:
        temp_pdf_path = os.path.join("/tmp", uploaded_file.name)
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.read())
        
        def extract_text_from_pdf(pdf_path):
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            
            chapters, articles = [], []
            current_chapter, current_chapter_content = "Uncategorized", []
            current_article, current_article_content = None, []
            paragraphs = text.split('\n')
            
            for para in paragraphs:
                para = para.strip()
                if re.match(r'^(Chapter (\d+|[A-Za-z]+)):.*$', para):
                    if current_chapter != "Uncategorized":
                        chapters.append({'title': current_chapter, 'content': ' '.join(current_chapter_content)})
                    current_chapter = para
                    current_chapter_content = []
                
                article_match = re.match(r'^(Article (\d+|[A-Za-z]+)):.*$', para)
                if article_match:
                    if current_article:
                        articles.append({'chapter': current_chapter, 'title': current_article, 'content': ' '.join(current_article_content)})
                    current_article = article_match.group(1)
                    current_article_content = []
                else:
                    if current_article:
                        current_article_content.append(para)
                    else:
                        current_chapter_content.append(para)
            
            if current_article:
                articles.append({'chapter': current_chapter, 'title': current_article, 'content': ' '.join(current_article_content)})
            if current_chapter and current_chapter != "Uncategorized":
                chapters.append({'title': current_chapter, 'content': ' '.join(current_chapter_content)})
            
            return chapters, articles

        def store_vectors(chapters, articles, pdf_name):
            namespace = pdf_name.replace(" ", "_").lower()  # Unique namespace for each PDF
            
            for i, chapter in enumerate(chapters):
                chapter_vector = model.encode(chapter['content']).tolist()
                index.upsert([(
                    f"{pdf_name}-chapter-{i}", chapter_vector, 
                    {"pdf_name": pdf_name, "text": chapter['content'], "type": "chapter"}
                )], namespace=namespace)  
            
            for i, article in enumerate(articles):
                article_number_match = re.search(r'Article (\d+|[A-Za-z]+)', article['title'], re.IGNORECASE)
                article_number = article_number_match.group(1) if article_number_match else str(i)
                article_vector = model.encode(article['content']).tolist()
                index.upsert([(
                    f"{pdf_name}-article-{article_number}", article_vector, 
                    {"pdf_name": pdf_name, "chapter": article['chapter'], "text": article['content'], "type": "article", "title": article['title"]}
                )], namespace=namespace)

        chapters, articles = extract_text_from_pdf(temp_pdf_path)
        store_vectors(chapters, articles, uploaded_file.name)
        selected_pdf = uploaded_file.name
        st.success(f"✅ PDF '{uploaded_file.name}' uploaded and stored in Pinecone!")

# Sidebar - Language Options
st.sidebar.title("🌍 Language Options")
input_lang = st.sidebar.radio("Choose Input Language", ["English", "Arabic"], index=0)
response_lang = st.sidebar.radio("Choose Response Language", ["English", "Arabic"], index=0)

# Query Input Section
st.title("⚖️ AI-Powered Legal HelpDesk")
query = st.text_input("Ask a legal question:")

# Function to Query Pinecone
def query_vectors(query, selected_pdf):
    query_vector = model.encode(query).tolist()
    namespace = selected_pdf.replace(" ", "_").lower()  # Get the correct namespace

    article_match = re.search(r'Article (\d+|[A-Za-z]+)', query, re.IGNORECASE)
    if article_match:
        article_number = article_match.group(1)
        results = index.query(
            vector=query_vector,
            top_k=1, 
            include_metadata=True, 
            filter={"pdf_name": {"$eq": selected_pdf}, "type": {"$eq": "article"}, "title": {"$eq": f"Article {article_number}"}},
            namespace=namespace
        )
        if results and results["matches"]:
            return results["matches"][0]["metadata"]["text"]
    
    results = index.query(
        vector=query_vector,
        top_k=5, 
        include_metadata=True, 
        filter={"pdf_name": {"$eq": selected_pdf}},
        namespace=namespace
    )
    
    if results and results["matches"]:
        return "\n\n".join([match["metadata"]["text"] for match in results["matches"]])
    
    return "No relevant answer found."

# Query Execution
if st.button("Get Answer"):
    if selected_pdf and query:
        response = query_vectors(query, selected_pdf)
        if response_lang == "Arabic":
            response = GoogleTranslator(source="auto", target="ar").translate(response)
            st.markdown(f"<div dir='rtl' style='text-align: right;'>{response}</div>", unsafe_allow_html=True)
        else:
            st.write(f"**Answer:** {response}")
    else:
        st.warning("Please select a PDF and enter a question.")
