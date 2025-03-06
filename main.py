import streamlit as st
import pinecone
import PyPDF2
import os
import re
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index_name = "helpdesk"
index = pc.Index(index_name)

# Load Hugging Face Model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Regex patterns for Chapters & Articles
chapter_pattern = r'^(Chapter (\d+|[A-Za-z]+)):.*$'
article_pattern = r'^(Article (\d+|[A-Za-z]+)):.*$'

def extract_text_from_pdf(pdf_path):
    """Extracts and structures text from the PDF."""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
    chapters, articles = [], []
    current_chapter, current_chapter_content = "Uncategorized", []
    current_article, current_article_content = None, []
    paragraphs = text.split('\n')
    
    for para in paragraphs:
        para = para.strip()
        
        if re.match(chapter_pattern, para):
            if current_chapter != "Uncategorized":
                chapters.append({'title': current_chapter, 'content': ' '.join(current_chapter_content)})
            current_chapter = para
            current_chapter_content = []
        
        article_match = re.match(article_pattern, para)
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
    """Stores extracted chapters and articles in Pinecone."""
    for i, chapter in enumerate(chapters):
        chapter_vector = model.encode(chapter['content']).tolist()
        index.upsert([(
            f"{pdf_name}-chapter-{i}", chapter_vector, 
            {"pdf_name": pdf_name, "text": chapter['content'], "type": "chapter"}
        )])
    
    for i, article in enumerate(articles):
        # Debugging: Print extracted article content
        st.write(f"Storing Article {article['title']}: {article['content'][:200]}...")  # Show first 200 characters for verification
        
        article_vector = model.encode(article['content']).tolist()
        index.upsert([(
            f"{pdf_name}-article-{i}", article_vector, 
            {"pdf_name": pdf_name, "chapter": article['chapter'], "text": article['content'], "type": "article"}
        )])

def query_vectors(query, selected_pdf):
    """Queries Pinecone for the most relevant result, prioritizing exact article matches."""
    query_vector = model.encode(query).tolist()
    
    # Exact search for Article 1
    if "article 1" in query.lower() or "article one" in query.lower():
        results = index.query(vector=query_vector, top_k=1, include_metadata=True, filter={"pdf_name": {"$eq": selected_pdf}, "type": {"$eq": "article"}, "title": {"$eq": "Article 1"}})
        if results and results["matches"]:
            return results["matches"][0]["metadata"]["text"]
    
    # Generic query for other articles or chapters
    results = index.query(vector=query_vector, top_k=5, include_metadata=True, filter={"pdf_name": {"$eq": selected_pdf}})
    
    if results and results["matches"]:
        return "\n\n".join([match["metadata"]["text"] for match in results["matches"]])
    return "No relevant answer found."

def translate_text(text, target_lang):
    """Translates text using GoogleTranslator."""
    return GoogleTranslator(source="auto", target=target_lang).translate(text)

# Streamlit UI
st.title("AI-Powered Legal HelpDesk")

# PDF Source Selection
pdf_source = st.radio("Select PDF Source", ["Upload from PC", "Choose from Document Storage"], index=1)
selected_pdf = None

if pdf_source == "Upload from PC":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file is not None:
        temp_pdf_path = os.path.join("/tmp", uploaded_file.name)
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.read())
        
        chapters, articles = extract_text_from_pdf(temp_pdf_path)
        store_vectors(chapters, articles, uploaded_file.name)
        selected_pdf = uploaded_file.name
        st.success("PDF uploaded and processed successfully!")
else:
    stored_pdfs = ["Basic Law Governance.pdf", "Law of the Consultative Council.pdf", "Law of the Council of Ministers.pdf"]
    selected_pdf = st.selectbox("Select a PDF", stored_pdfs)

# Language Selection
input_lang = st.radio("Choose Input Language", ["English", "Arabic"], index=0)
response_lang = st.radio("Choose Response Language", ["English", "Arabic"], index=0)

# Query Input
query = st.text_input("Ask a legal question:")

if st.button("Get Answer"):
    if selected_pdf and query:
        response = query_vectors(query, selected_pdf)
        if response_lang == "Arabic":
            response = translate_text(response, "ar")
            st.markdown(f"<div dir='rtl' style='text-align: right;'>{response}</div>", unsafe_allow_html=True)
        else:
            st.write(f"**Answer:** {response}")
    else:
        st.warning("Please upload a PDF and enter a query.")
