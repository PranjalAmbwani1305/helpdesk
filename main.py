import os
import re
import streamlit as st
import fitz  # PyMuPDF for PDF processing
import pinecone
import hashlib
import asyncio
import torch
from transformers import AutoTokenizer, AutoModel
from deep_translator import GoogleTranslator  # Translation Support

# 🔹 Set up Pinecone API
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
index_name = "helpdesk"

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(index_name)

# ✅ Ensure Async Event Loop Setup
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# 🔍 Load Hugging Face Model for Text Embeddings
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# 🌍 Translation Function
def translate_text(text, target_lang):
    return GoogleTranslator(source="auto", target=target_lang).translate(text)

# 🎯 Function to generate text embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy().tolist()  # Convert to list

# 📜 Function to extract articles from PDF
def extract_articles_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""

    for page in doc:
        full_text += page.get_text("text") + "\n"

    if not full_text.strip():
        return None  # If no text is extracted, return None

    # 🔹 Article Detection (Supports "Article X", "Section X", "Chapter X", or "1. Title")
    article_pattern = re.compile(
        r'(?i)\b(Article\s\d+|Section\s\d+|Chapter\s\d+|\n\d+\.\s[A-Za-z])', re.IGNORECASE
    )

    # 🔹 Splitting based on detected pattern
    articles = re.split(article_pattern, full_text)

    if len(articles) < 2:  # If no valid split occurs, return full text as one article
        return [full_text.strip()]

    return [article.strip() for article in articles if article.strip()]

# 📂 Function to upload and store PDFs in Pinecone (article-wise)
def process_and_store_pdf(uploaded_file):
    if uploaded_file is not None:
        pdf_name = uploaded_file.name.replace(" ", "_").lower()
        file_path = os.path.join("/tmp", pdf_name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        articles = extract_articles_from_pdf(file_path)

        if not articles:
            st.error("⚠️ No articles detected in the PDF.")
            return

        pdf_id = hashlib.md5(pdf_name.encode()).hexdigest()

        for idx, article in enumerate(articles):
            article_id = f"{pdf_id}_article_{idx}"
            vector = get_embedding(article)

            index.upsert(vectors=[(article_id, vector, {"pdf_name": pdf_name, "content": article})], namespace=pdf_name)

        st.success(f"✅ PDF '{pdf_name}' processed successfully with {len(articles)} articles!")

# 🔹 Query Function
def query_pinecone(query_text, selected_pdf):
    query_vector = get_embedding(query_text)

    try:
        results = index.query(
            namespace=selected_pdf,
            queries=[query_vector],  # Ensure it's a list
            top_k=5,
            include_metadata=True
        )

        return results["matches"] if results["matches"] else []
    
    except Exception as e:
        st.error(f"⚠️ Pinecone query failed: {str(e)}")
        return []

# 🎨 UI: Sidebar for PDF Upload & Language Selection
st.sidebar.title("📂 Upload Legal Document")
uploaded_file = st.sidebar.file_uploader("📂 Upload PDF (Limit 200MB)", type=["pdf"])

if uploaded_file:
    process_and_store_pdf(uploaded_file)

# 🌍 Language Selection
language = st.sidebar.radio("🌍 Select Language", ["English", "Arabic"])

# 🎨 UI: Main Page
st.markdown(f"<h1 style='text-align: center;'>📜 AI-Powered Legal HelpDesk ({'English' if language == 'English' else 'عربي'})</h1>", unsafe_allow_html=True)

# 🔍 Query Section
st.subheader("🤖 Ask a Legal Question" if language == "English" else "🤖 اسأل سؤال قانوني")
query = st.text_area("✍️ Type your question here:" if language == "English" else "✍️ اكتب سؤالك هنا:")

if st.button("🔎 Get Answer" if language == "English" else "🔎 احصل على الإجابة"):
    stored_namespaces = index.describe_index_stats().get("namespaces", {}).keys()
    if not stored_namespaces:
        st.error("⚠️ No PDFs available. Upload a document first.")
    else:
        translated_query = translate_text(query, "en") if language == "Arabic" else query
        all_results = []

        for pdf_namespace in stored_namespaces:
            all_results.extend(query_pinecone(translated_query, pdf_namespace))

        if not all_results:
            st.error("⚠️ No relevant articles found.")
        else:
            best_match = all_results[0]["metadata"]["content"]
            translated_answer = translate_text(best_match, "ar") if language == "Arabic" else best_match

            st.markdown("### ✅ AI Answer:" if language == "English" else "### ✅ إجابة الذكاء الاصطناعي:")
            st.info(translated_answer)
