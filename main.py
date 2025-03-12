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
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# 📜 Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

# 📂 Function to upload and store multiple PDFs in Pinecone
def process_and_store_pdf(uploaded_file):
    if uploaded_file is not None:
        pdf_name = uploaded_file.name
        file_path = os.path.join("/tmp", pdf_name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        pdf_text = extract_text_from_pdf(file_path)

        # Generate unique ID using hash
        pdf_id = hashlib.md5(pdf_name.encode()).hexdigest()
        vector = get_embedding(pdf_text)

        # Store in Pinecone
        index.upsert(vectors=[(pdf_id, vector, {"pdf_name": pdf_name, "content": pdf_text})])

        st.success(f"✅ PDF '{pdf_name}' uploaded and stored successfully!")

# 📑 Function to get stored PDFs (Updated Query)
def get_stored_pdfs():
    try:
        index_stats = index.describe_index_stats()
        pdf_names = set()

        if "namespaces" in index_stats and index_stats["total_vector_count"] > 0:
            for namespace in index_stats["namespaces"]:
                if index_stats["namespaces"][namespace]["vector_count"] > 0:
                    query_vector = get_embedding("dummy query")  # Valid embedding
                    query_results = index.query(
                        namespace=namespace,
                        queries=[query_vector],  
                        top_k=5,
                        include_metadata=True
                    )

                    for match in query_results.get("matches", []):
                        if "metadata" in match and "pdf_name" in match["metadata"]:
                            pdf_names.add(match["metadata"]["pdf_name"])

        return list(pdf_names)

    except Exception as e:
        st.error(f"⚠️ Pinecone query failed: {str(e)}")
        return []

# 🎨 UI: Sidebar for Uploaded PDFs
st.sidebar.title("📂 Stored PDFs")
stored_pdfs = get_stored_pdfs()
selected_pdf = st.sidebar.selectbox("📜 Select a PDF", stored_pdfs if stored_pdfs else ["No PDFs Found"], key="pdf_dropdown_unique")

# 🌍 Language Selection
language = st.sidebar.radio("🌍 Select Language", ["English", "Arabic"])

# 🎨 UI: Main Page
st.markdown(f"<h1 style='text-align: center;'>📜 AI-Powered Legal HelpDesk ({'English' if language == 'English' else 'عربي'})</h1>", unsafe_allow_html=True)

# 🔹 PDF Upload Section
st.subheader("📑 Select PDF Source" if language == "English" else "📑 اختر مصدر ملف PDF")
upload_option = st.radio("Choose an option:" if language == "English" else "اختر خيارًا:", ["Upload from PC" if language == "English" else "تحميل من الكمبيوتر", "Choose from Document Storage" if language == "English" else "اختر من التخزين"])

if upload_option == "Upload from PC" or upload_option == "تحميل من الكمبيوتر":
    uploaded_file = st.file_uploader("📂 Upload a PDF" if language == "English" else "📂 تحميل ملف PDF", type=["pdf"], key="pdf_uploader")
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
            # Query Pinecone
            results = index.query(queries=[query_vector], top_k=5, include_metadata=True)
            if results["matches"]:
                answer = results["matches"][0]["metadata"].get("content", "⚠️ No relevant information found.")
            else:
                answer = "⚠️ No relevant information found."

            translated_answer = translate_text(answer, "ar") if language == "Arabic" else answer

            st.markdown("### ✅ AI Answer:" if language == "English" else "### ✅ إجابة الذكاء الاصطناعي:")
            st.info(translated_answer)

        except Exception as e:
            st.error(f"❌ Error fetching results: {str(e)}")
    else:
        st.error("⚠️ Please select a PDF before asking a question." if language == "English" else "⚠️ يرجى تحديد ملف PDF قبل طرح سؤال.")                     
