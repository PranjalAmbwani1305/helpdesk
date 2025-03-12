import os
import streamlit as st
import fitz  # PyMuPDF
import pinecone
import hashlib
import asyncio
import torch
import re
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

# 📜 Function to extract and chunk text from PDF
def extract_and_chunk_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    chunks = []
    
    for page in doc:
        text = page.get_text("text")
        articles = re.split(r'Article\s+\d+', text)  # Split by "Article X"
        article_numbers = re.findall(r'Article\s+\d+', text)  # Extract article numbers

        for i, article_text in enumerate(articles[1:], start=0):  # Ignore first empty split
            chunk = {
                "id": hashlib.md5(f"{pdf_path}-article-{i}".encode()).hexdigest(),
                "title": article_numbers[i] if i < len(article_numbers) else f"Section {i+1}",
                "text": article_text.strip(),
                "type": "article"
            }
            chunks.append(chunk)

    return chunks

# 📂 Function to upload and store PDF chunks in Pinecone
def process_and_store_pdf(uploaded_file):
    if uploaded_file is not None:
        pdf_name = uploaded_file.name.replace(" ", "_").lower()
        file_path = os.path.join("/tmp", pdf_name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        pdf_chunks = extract_and_chunk_pdf(file_path)

        for chunk in pdf_chunks:
            vector = get_embedding(chunk["text"])
            index.upsert(vectors=[(chunk["id"], vector, {"title": chunk["title"], "text": chunk["text"], "pdf_name": pdf_name})], namespace=pdf_name)

        st.success(f"✅ PDF '{pdf_name}' uploaded and processed into chunks!")

# 📑 Function to get available namespaces
def get_stored_namespaces():
    try:
        index_stats = index.describe_index_stats()
        if "namespaces" in index_stats:
            return list(index_stats["namespaces"].keys())
    except Exception as e:
        st.error(f"⚠️ Pinecone error: {str(e)}")
    return []

# 🎨 UI: Sidebar for Available Namespaces
st.sidebar.title("📂 Available PDFs (Namespaces)")
stored_namespaces = get_stored_namespaces()
selected_namespace = st.sidebar.selectbox("📜 Select a PDF Namespace", stored_namespaces if stored_namespaces else ["No PDFs Found"])

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
    if selected_namespace and selected_namespace != "No PDFs Found":
        translated_query = translate_text(query, "en") if language == "Arabic" else query
        query_vector = get_embedding(translated_query)

        try:
            # Query Pinecone with selected namespace
            results = index.query(
                namespace=selected_namespace,
                queries=[query_vector],  # Ensure it's a list
                top_k=5,
                include_metadata=True
            )

            if results["matches"]:
                best_match = results["matches"][0]["metadata"]
                answer_text = best_match["text"]

                translated_answer = translate_text(answer_text, "ar") if language == "Arabic" else answer_text

                st.markdown("### ✅ AI Answer:" if language == "English" else "### ✅ إجابة الذكاء الاصطناعي:")
                st.info(translated_answer)
            else:
                st.warning("⚠️ No relevant information found." if language == "English" else "⚠️ لم يتم العثور على معلومات ذات صلة.")
        
        except Exception as e:
            st.error(f"⚠️ Pinecone query failed: {str(e)}")

    else:
        st.error("⚠️ Please select a PDF namespace before asking a question." if language == "English" else "⚠️ يرجى تحديد مساحة اسم ملف PDF قبل طرح سؤال.")  
