import os
import pinecone
import streamlit as st
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

# ✅ Initialize Pinecone Client
pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = "helpdesk"

# ✅ Ensure the index exists
if INDEX_NAME not in pc.list_indexes():
    st.error(f"⚠️ Pinecone index '{INDEX_NAME}' not found. Please create it in Pinecone console.")
else:
    index = pc.Index(INDEX_NAME)

# ✅ Load SentenceTransformer model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ✅ Function to extract text from PDFs
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    return text

# ✅ Function to store PDF content in Pinecone
def store_pdf_in_pinecone(pdf_name, pdf_text):
    vector = embedding_model.encode(pdf_text).tolist()
    index.upsert([(pdf_name, vector, {"filename": pdf_name})])

# ✅ Streamlit UI Setup
st.set_page_config(page_title="Legal HelpDesk", page_icon="⚖️", layout="wide")

# ✅ Sidebar with Options
with st.sidebar:
    st.title("⚖️ Legal HelpDesk")
    st.info("Helping you find legal information from Saudi Arabian laws quickly and accurately.")
    
    # 🌍 Language Selection
    input_language = st.radio("🌍 Choose Input Language:", ["English", "Arabic"])
    response_language = st.radio("🌍 Choose Response Language:", ["English", "Arabic"])

# ✅ Main Title
st.title("AI-Powered Legal HelpDesk for Saudi Arabia")

# 📂 PDF Source Selection
pdf_source = st.radio("📂 Select PDF Source:", ["Upload from PC", "Choose from Document Storage"])

# ✅ File Upload Section
if pdf_source == "Upload from PC":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        with st.spinner("Processing PDF..."):
            pdf_text = extract_text_from_pdf(uploaded_file)
            store_pdf_in_pinecone(uploaded_file.name, pdf_text)
        st.success(f"✅ {uploaded_file.name} stored successfully!")

# ✅ Stored PDFs Section
elif pdf_source == "Choose from Document Storage":
    try:
        st.subheader("📄 Stored Legal Documents")
        total_pdfs = index.describe_index_stats()["total_vector_count"]
        st.write(f"📄 **Total PDFs Stored:** {total_pdfs}")
    except Exception as e:
        st.error(f"⚠️ Error fetching stored PDFs: {e}")

# 🔍 Search Bar for Queries
query = st.text_input("🔍 Ask a legal question:")

# ✅ Query Processing
if query:
    with st.spinner("Searching legal documents..."):
        query_vector = embedding_model.encode(query).tolist()
        results = index.query(query_vector, top_k=5, include_metadata=True)

    # ✅ Display Results
    st.subheader("📜 Relevant Legal Documents:")
    if "matches" in results and results["matches"]:
        for match in results["matches"]:
            st.write(f"📄 {match['metadata']['filename']} (Score: {match['score']:.2f})")
    else:
        st.warning("⚠️ No relevant legal documents found.")

# ✅ Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>Developed with ❤️ using Streamlit & Pinecone</p>", unsafe_allow_html=True)
