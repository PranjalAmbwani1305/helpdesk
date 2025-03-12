import os
import re
import streamlit as st
import fitz  # PyMuPDF
import pinecone
import hashlib
import asyncio
import torch
from transformers import AutoTokenizer, AutoModel
from deep_translator import GoogleTranslator  # Translation Support

# ✅ Set up Pinecone API
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
index_name = "helpdesk"

if not PINECONE_API_KEY:
    st.error("⚠️ Pinecone API Key is missing! Set it in environment variables.")

# 🔹 Initialize Pinecone
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

# 🎯 Generate text embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy().tolist()  # Convert to list

# 📜 Extract text from PDF and split into articles
def extract_articles_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    
    for page in doc:
        full_text += page.get_text("text") + "\n"

    # 📝 Split text into articles based on "Article", "Section", or numbering like 1., 2., etc.
    article_pattern = re.compile(r'(Article\s\d+|Section\s\d+|\n\d+\.)', re.IGNORECASE)
    articles = re.split(article_pattern, full_text)

    # Filter out empty articles
    return [article.strip() for article in articles if article.strip()]

# 📂 Upload and store PDFs in Pinecone (Article-wise)
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

        # 🔹 Store each article separately
        vectors = []
        for idx, article in enumerate(articles):
            article_id = hashlib.md5((pdf_name + str(idx)).encode()).hexdigest()
            vector = get_embedding(article)

            vectors.append((article_id, vector, {"pdf_name": pdf_name, "article": article}))

        index.upsert(vectors=vectors, namespace=pdf_name)
        st.success(f"✅ PDF '{pdf_name}' uploaded with {len(articles)} articles stored!")

# 📑 Get available PDFs
def get_stored_pdfs():
    try:
        index_stats = index.describe_index_stats()
        
        if "namespaces" in index_stats:
            return list(index_stats["namespaces"].keys())
    except Exception as e:
        st.error(f"⚠️ Pinecone error: {str(e)}")
    return []

# 🎨 UI: Sidebar for Available PDFs
st.sidebar.title("📂 Available PDFs")
stored_pdfs = get_stored_pdfs()
selected_pdf = st.sidebar.selectbox("📜 Select a PDF", stored_pdfs if stored_pdfs else ["No PDFs Found"])

# 🌍 Language Selection
language = st.sidebar.radio("🌍 Select Language", ["English", "Arabic"])

# 🎨 UI: Main Page
st.markdown(f"<h1 style='text-align: center;'>📜 AI-Powered Legal HelpDesk ({'English' if language == 'English' else 'عربي'})</h1>", unsafe_allow_html=True)

# 🔹 PDF Upload Section
st.subheader("📑 Select PDF Source" if language == "English" else "📑 اختر مصدر ملف PDF")
uploaded_file = st.file_uploader("📂 Upload a PDF" if language == "English" else "📂 تحميل ملف PDF", type=["pdf"])
if uploaded_file:
    process_and_store_pdf(uploaded_file)

# 🔍 Query Section
st.subheader("🤖 Ask a Legal Question" if language == "English" else "🤖 اسأل سؤال قانوني")
query = st.text_area("✍️ Type your question here:" if language == "English" else "✍️ اكتب سؤالك هنا:")

if st.button("🔎 Get Answer" if language == "English" else "🔎 احصل على الإجابة"):
    if selected_pdf and selected_pdf != "No PDFs Found":
        translated_query = translate_text(query, "en") if language == "Arabic" else query
        query_vector = get_embedding(translated_query)

        try:
            # Query Pinecone with selected PDF
            results = index.query(
                namespace=selected_pdf,
                queries=[query_vector],  # Ensure it's a list
                top_k=3,
                include_metadata=True
            )

            if results["matches"]:
                answer = results["matches"][0]["metadata"]["article"]
            else:
                answer = "⚠️ No relevant information found." if language == "English" else "⚠️ لم يتم العثور على معلومات ذات صلة."

            translated_answer = translate_text(answer, "ar") if language == "Arabic" else answer

            st.markdown("### ✅ AI Answer:" if language == "English" else "### ✅ إجابة الذكاء الاصطناعي:")
            st.info(translated_answer)
        
        except Exception as e:
            st.error(f"⚠️ Pinecone query failed: {str(e)}")

    else:
        st.error("⚠️ Please select a PDF before asking a question." if language == "English" else "⚠️ يرجى تحديد ملف PDF قبل طرح سؤال.")     
