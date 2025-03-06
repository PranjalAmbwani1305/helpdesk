import streamlit as st
import pinecone
import PyPDF2
import os
import re
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables for Pinecone API key
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
index_name = "helpdesk"

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # Adjust based on your embedding model
        metric="cosine"
    )
index = pc.Index(index_name)

# Initialize sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Regex patterns for Chapters & Articles
chapter_pattern = r'^(Chapter (\d+|[A-Za-z]+)):.*$'
article_pattern = r'^(Article (\d+|[A-Za-z]+)):.*$'

def list_stored_pdfs():
    return ["Basic Law Governance.pdf", "Law of the Consultative Council.pdf", "Law of the Council of Ministers.pdf"]

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>AI-Powered Legal HelpDesk for Saudi Arabia</h1>", unsafe_allow_html=True)

# Sidebar for stored PDFs
st.sidebar.header("📂 Stored PDFs")
pdf_list = list_stored_pdfs()
if pdf_list:
    with st.sidebar.expander("📜 View Stored PDFs", expanded=False):
        for pdf in pdf_list:
            st.sidebar.write(f"📄 {pdf}")
else:
    st.sidebar.write("No PDFs stored yet. Upload one!")

selected_pdf = None
pdf_source = st.radio("Select PDF Source", ["Upload from PC", "Choose from the Document Storage"], key="pdf_source")

if pdf_source == "Upload from PC":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], key="file_upload")
    if uploaded_file:
        temp_pdf_path = f"temp_{uploaded_file.name}"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success("PDF uploaded successfully!")
else:
    selected_pdf = st.selectbox("Choose from stored documents", pdf_list, key="stored_pdf")

# Language Selection
input_language = st.selectbox("Choose Input Language", ("English", "Arabic"), key="input_lang")
response_language = st.selectbox("Choose Response Language", ("English", "Arabic"), key="response_lang")

# Query Input
gpt_query = st.text_input("Ask a question (in English or Arabic):", key="user_query")

if st.button("Get Answer", key="query_button"):
    if gpt_query:
        # Convert user query to embedding
        query_embedding = model.encode(gpt_query).tolist()

        # Search Pinecone index for relevant passage
        search_result = index.query(vector=query_embedding, top_k=3, include_metadata=True)

        if search_result and 'matches' in search_result:
            if search_result["matches"]:
                best_match = search_result["matches"][0]
                retrieved_text = best_match["metadata"]["text"]
                st.write("**Answer:**", retrieved_text)
            else:
                st.write("**Answer:** No relevant information found.")
        else:
            st.write("**Answer:** No relevant information found.")
    else:
        st.warning("Please enter a question.")
