import os
import streamlit as st
import fitz  # PyMuPDF
import pinecone
import hashlib
import asyncio
import torch
from transformers import AutoTokenizer, AutoModel
from deep_translator import GoogleTranslator  # Translation Support

# 🌟 Set up Pinecone API
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

# 📜 Function to extract and chunk text by article from PDF
def extract_articles_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    articles = []
    current_article = ""

    for page in doc:
        text = page.get_text("text")
        for line in text.split("\n"):
            if line.lower().startswith("article") or line.lower().startswith("section"):
                if current_article:
                    articles.append(current_article.strip())
                current_article = line
            else:
                current_article += " " + line
        
    if current_article:
        articles.append(current_article.strip())

    return articles if articles else None

# 📂 Function to upload and store PDFs in Pinecone (article-wise)
def process_and_store_pdf(uploaded_file):
    if uploaded_file is not None:
        pdf_name = uploaded_file.name.replace(" ", "_").lower()
        file_path = os.path.join("/tmp", pdf_name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        articles = extract_articles_from_pdf(file_path)

        if not articles:
            st.warning("⚠️ No articles detected in the PDF.")
            return

        for idx, article in enumerate(articles):
            article_id = hashlib.md5(f"{pdf_name}_{idx}".encode()).hexdigest()
            vector = get_embedding(article)

            index.upsert(
                vectors=[(article_id, vector, {"pdf_name": pdf_name, "content": article})],
                namespace=pdf_name
            )

        st.success(f"✅ PDF '{pdf_name}' processed and stored with {len(articles)} articles!")

# 🎨 UI: Main Page
st.markdown("<h1 style='text-align: center;'>📜 AI-Powered Legal HelpDesk</h1>", unsafe_allow_html=True)

# 📂 PDF Selection
st.subheader("📂 Select PDF Source")
col1, col2 = st.columns(2)
with col1:
    st.button("📥 Upload from PC")
with col2:
    st.button("📂 Choose from Document Storage")

# 📑 File Upload
uploaded_file = st.file_uploader("📂 Upload PDF", type=["pdf"])
st.markdown("**Limit: 200MB per file**")

if uploaded_file:
    process_and_store_pdf(uploaded_file)

# 🌍 Language Selection
st.subheader("🌍 Choose Input Language")
input_language = st.radio("Select Input Language", ["English", "Arabic"], key="input_lang")

st.subheader("🌍 Choose Response Language")
response_language = st.radio("Select Response Language", ["English", "Arabic"], key="response_lang")

# 🔍 Query Section
st.subheader("🤖 Ask a Legal Question")
query = st.text_area("✍️ Type your question here:")

if st.button("🔎 Get Answer"):
    if uploaded_file:
        translated_query = translate_text(query, "en") if input_language == "Arabic" else query
        query_vector = get_embedding(translated_query)

        try:
            # Query Pinecone with selected namespace
            results = index.query(
                namespace=uploaded_file.name.replace(" ", "_").lower(),
                queries=[query_vector],  
                top_k=5,
                include_metadata=True
            )

            answer = results["matches"][0]["metadata"]["content"] if results["matches"] else "⚠️ No relevant information found."

            translated_answer = translate_text(answer, "ar") if response_language == "Arabic" else answer

            st.markdown("### ✅ AI Answer:")
            st.info(translated_answer)

        except Exception as e:
            st.error(f"⚠️ Pinecone query failed: {str(e)}")

    else:
        st.error("⚠️ Please upload a PDF before asking a question.")
