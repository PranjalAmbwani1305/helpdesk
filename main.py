import os
import streamlit as st
import fitz  # PyMuPDF
import pinecone
import hashlib
import torch
from transformers import AutoTokenizer, AutoModel
from deep_translator import GoogleTranslator

# ✅ Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
index_name = "helpdesk"

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(index_name)

# ✅ Load Hugging Face Model
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
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# 📜 Extract text from PDF article-wise
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    articles = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            articles.append((page_num, text))
    return articles

# 📂 Store PDF articles in Pinecone
def process_and_store_pdf(uploaded_file):
    if uploaded_file is not None:
        pdf_name = uploaded_file.name
        file_path = os.path.join("/tmp", pdf_name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        articles = extract_text_from_pdf(file_path)

        # 🔹 Store each article separately
        vectors = []
        for page_num, article_text in articles:
            doc_id = hashlib.md5(f"{pdf_name}_{page_num}".encode()).hexdigest()
            vectors.append((doc_id, get_embedding(article_text), {"pdf_name": pdf_name, "content": article_text}))

        if vectors:
            index.upsert(vectors=vectors)
            st.success(f"✅ PDF '{pdf_name}' uploaded and stored successfully!")

# 📑 Fetch unique PDF names
def get_stored_pdfs():
    query_results = index.describe_index_stats()
    pdf_names = set()

    if "namespaces" in query_results and query_results["total_vector_count"] > 0:
        for namespace in query_results["namespaces"]:
            vectors = index.query(namespace=namespace, queries=[[0]*384], top_k=50, include_metadata=True)
            for match in vectors["matches"]:
                pdf_names.add(match["metadata"]["pdf_name"])

    return list(pdf_names)

# 🎨 UI: Sidebar for Stored PDFs
st.sidebar.title("📂 Stored PDFs")
stored_pdfs = get_stored_pdfs()
selected_pdf = st.sidebar.selectbox("📜 Select a PDF", stored_pdfs if stored_pdfs else ["No PDFs Found"])

# 🌍 Language Selection
language = st.sidebar.radio("🌍 Select Language", ["English", "Arabic"])

# 🎨 UI: Main Page
st.markdown(f"<h1 style='text-align: center;'>📜 AI-Powered Legal HelpDesk ({'English' if language == 'English' else 'عربي'})</h1>", unsafe_allow_html=True)

# 🔹 PDF Upload Section
st.subheader("📑 Upload PDFs")
uploaded_file = st.file_uploader("📂 Upload a PDF", type=["pdf"])
if uploaded_file:
    process_and_store_pdf(uploaded_file)

# 🔍 Query Section
st.subheader("🤖 Ask a Legal Question")
query = st.text_area("✍️ Type your question here:")

if st.button("🔎 Get Answer"):
    if selected_pdf and selected_pdf != "No PDFs Found":
        translated_query = translate_text(query, "en") if language == "Arabic" else query
        query_vector = get_embedding(translated_query)

        # Query Pinecone for all articles in the selected PDF
        results = index.query(queries=[query_vector], top_k=5, include_metadata=True, filter={"pdf_name": selected_pdf})
        answers = [match["metadata"]["content"] for match in results["matches"] if "metadata" in match]

        final_answer = "\n\n".join(answers) if answers else "⚠️ No relevant information found."
        translated_answer = translate_text(final_answer, "ar") if language == "Arabic" else final_answer

        st.markdown("### ✅ AI Answer:")
        st.info(translated_answer)
    else:
        st.error("⚠️ Please select a PDF before asking a question.")
