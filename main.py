import os
import streamlit as st
import fitz  # PyMuPDF for PDF processing
import pinecone
import hashlib
import asyncio
import torch
from transformers import AutoTokenizer, AutoModel
from deep_translator import GoogleTranslator

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
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy().tolist()

# 📜 Extract text from PDF
def extract_articles_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])

    return text.split("\n\n") if text.strip() else None

# 📂 Function to upload & store PDFs
def process_and_store_pdf(uploaded_file):
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

    st.success(f"✅ PDF '{pdf_name}' processed with {len(articles)} articles!")

# 🔹 Query Function
def query_pinecone(query_text, selected_pdf):
    query_vector = get_embedding(query_text)
    results = index.query(namespace=selected_pdf, queries=[query_vector], top_k=5, include_metadata=True)
    return results["matches"] if results["matches"] else []

# 🎨 **Streamlit UI**
st.sidebar.title("📂 Document Management")

# 📂 **Stored PDFs Section**
st.sidebar.subheader("📂 Stored PDFs")
stored_namespaces = list(index.describe_index_stats().get("namespaces", {}).keys())

if stored_namespaces:
    for pdf in stored_namespaces:
        st.sidebar.write(f"📄 {pdf.replace('_', ' ').title()}")
else:
    st.sidebar.write("⚠️ No PDFs stored.")

# 📂 **Upload or Choose Source**
st.sidebar.subheader("📄 Select PDF Source")
source_option = st.sidebar.radio("", ["Upload from PC", "Choose from Document Storage"])

if source_option == "Upload from PC":
    uploaded_file = st.sidebar.file_uploader("📂 Upload PDF", type=["pdf"])
    if uploaded_file:
        process_and_store_pdf(uploaded_file)

elif source_option == "Choose from Document Storage":
    selected_pdf = st.sidebar.selectbox("📜 Select a PDF", stored_namespaces) if stored_namespaces else None

# 🌍 **Language Selection**
st.sidebar.subheader("🌍 Choose Input Language")
input_language = st.sidebar.radio("", ["English", "Arabic"])

st.sidebar.subheader("🌍 Choose Response Language")
response_language = st.sidebar.radio("", ["English", "Arabic"])

# 🎨 **Main UI**
st.markdown("<h1 style='text-align: center;'>📜 AI-Powered Legal HelpDesk for Saudi Arabia</h1>", unsafe_allow_html=True)

# 🔍 **Query Section**
query = st.text_area("✍️ Ask a question (in English or Arabic):")

if st.button("🔎 Get Answer"):
    if not stored_namespaces:
        st.error("⚠️ No PDFs available. Upload a document first.")
    else:
        translated_query = translate_text(query, "en") if input_language == "Arabic" else query
        results = query_pinecone(translated_query, selected_pdf if source_option == "Choose from Document Storage" else None)

        if not results:
            st.error("⚠️ No relevant articles found.")
        else:
            best_match = results[0]["metadata"]["content"]
            translated_answer = translate_text(best_match, "ar") if response_language == "Arabic" else best_match

            st.markdown("### ✅ AI Answer:")
            st.info(translated_answer)
