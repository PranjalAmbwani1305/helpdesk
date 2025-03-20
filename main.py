import os
import pinecone

import streamlit as st
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

# 🌍 Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "helpdesk"

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# ⚡ Load Embedding Model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 📄 Extract Text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text + "\n"
    return text

# 🗄️ Store PDF in Pinecone
def store_pdf_in_pinecone(pdf_name, pdf_text):
    try:
        vector = embedding_model.encode(pdf_text).tolist()
        index.upsert(vectors=[{"id": pdf_name, "values": vector, "metadata": {"filename": pdf_name}}])
        return True
    except Exception as e:
        st.error(f"Error storing PDF: {e}")
        return False

# 📑 Fetch Stored PDFs
def get_stored_pdfs():
    try:
        response = index.describe_index_stats()
        total_vectors = response.get("total_vector_count", 0)

        if total_vectors == 0:
            return []

        query_response = index.query(vector=[0.0] * 384, top_k=total_vectors, include_metadata=True)

        filenames = []
        for match in query_response.get("matches", []):
            raw_filename = match.get("metadata", {}).get("filename", "")
            if raw_filename:
                cleaned_filename = raw_filename.replace(".pdf", "").split("-article")[0]
                filenames.append(f"📑 {cleaned_filename}")

        return list(set(filenames))  # Remove duplicates
    except Exception as e:
        st.error(f"Error fetching stored PDFs: {e}")
        return []

# 🌍 **Language Selection**
st.sidebar.markdown("### 🌍 **Choose Input Language | اختر لغة الإدخال**")
input_language = st.sidebar.radio("", ["English", "العربية"])

st.sidebar.markdown("### 🌍 **Choose Response Language | اختر لغة الرد**")
response_language = st.sidebar.radio("", ["English", "العربية"])

# 🌟 **Set UI Labels Based on Language**
if input_language == "English":
    page_title = "⚖️ AI-Powered Legal HelpDesk for Saudi Arabia"
    upload_label = "📌 Upload from PC"
    file_upload_msg = "📤 Upload a PDF"
    stored_label = "📂 Choose from Document Storage"
    success_msg = "✅ File successfully stored!"
    stored_files_label = "📚 Stored Legal Documents"
    no_files_msg = "🚫 No PDFs found in storage."
    select_file_msg = "🔍 Select a document:"
    selected_file_msg = "📖 You selected: **{}**"
    ask_question_label = "📝 Ask a question (in English or Arabic):"
    submit_btn = "Submit"
else:  # Arabic
    page_title = "⚖️ المساعد القانوني الذكي للسعودية"
    upload_label = "📌 تحميل من الكمبيوتر"
    file_upload_msg = "📤 تحميل ملف PDF"
    stored_label = "📂 اختر من التخزين"
    success_msg = "✅ تم تخزين الملف بنجاح!"
    stored_files_label = "📚 المستندات القانونية المخزنة"
    no_files_msg = "🚫 لا يوجد ملفات PDF مخزنة."
    select_file_msg = "🔍 اختر مستندًا:"
    selected_file_msg = "📖 لقد اخترت: **{}**"
    ask_question_label = "📝 اطرح سؤالًا (بالإنجليزية أو العربية):"
    submit_btn = "إرسال"

# 🎨 **Streamlit UI Layout**
st.set_page_config(page_title=page_title, page_icon="⚖️", layout="wide")

# 🏛️ **Header**
st.markdown(f"<h1 style='text-align: center;'>{page_title}</h1>", unsafe_allow_html=True)

# 📂 **PDF Upload Section**
st.subheader(upload_label)
uploaded_file = st.file_uploader(file_upload_msg, type=["pdf"])
if uploaded_file:
    pdf_text = extract_text_from_pdf(uploaded_file)
    success = store_pdf_in_pinecone(uploaded_file.name, pdf_text)
    if success:
        st.success(success_msg)

# 📁 **Stored PDFs**
st.subheader(stored_label)
stored_pdfs = get_stored_pdfs()

if stored_pdfs:
    selected_pdf = st.selectbox(select_file_msg, stored_pdfs)
    st.write(selected_file_msg.format(selected_pdf))
else:
    st.info(no_files_msg)

# 💡 **Legal Question Input**
st.subheader(ask_question_label)
user_question = st.text_area("", placeholder="Type your legal question here...")

if st.button(submit_btn):
    if not user_question.strip():
        st.warning("⚠️ Please enter a question.")
    else:
        st.success(f"✅ Your question has been submitted: {user_question}")
