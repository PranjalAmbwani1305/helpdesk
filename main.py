import streamlit as st
import pinecone
import tempfile
import os

# Initialize Pinecone with environment variable
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "helpdesk"

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Function to retrieve stored PDFs from Pinecone
def get_stored_pdfs():
    stored_pdfs = []
    query_result = index.query(vector=[0] * 512, top_k=10, include_metadata=True)  # Dummy vector query
    for match in query_result['matches']:
        if 'filename' in match['metadata']:
            stored_pdfs.append(match['metadata']['filename'])
    return stored_pdfs

# Streamlit UI
st.set_page_config(page_title="AI-Powered Legal HelpDesk", layout="wide")

st.title("🛡️ AI-Powered Legal HelpDesk for Saudi Arabia")

# Sidebar - Stored PDFs
st.sidebar.title("📂 Stored PDFs")
stored_pdfs = get_stored_pdfs()

if stored_pdfs:
    for pdf in stored_pdfs:
        st.sidebar.write(f"📄 {pdf}")
else:
    st.sidebar.write("No PDFs stored yet.")

# PDF Upload Options
st.subheader("Select PDF Source")
pdf_source = st.radio("Select PDF Source", ["Upload from PC", "Choose from the Document Storage"])

if pdf_source == "Upload from PC":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    
    if uploaded_file:
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"✅ {uploaded_file.name} uploaded successfully!")

# Language Selection
st.subheader("Choose Input Language")
input_language = st.radio("Choose Input Language", ["English", "Arabic"])

st.subheader("Choose Response Language")
response_language = st.radio("Choose Response Language", ["English", "Arabic"])

# Question Input
st.subheader("Ask a question (in English or Arabic):")
user_query = st.text_input("Enter your legal query...")

# Process Query (Placeholder)
if st.button("Submit"):
    if user_query:
        st.success("🔍 Searching for relevant legal information...")
        # Here, you would integrate LLM-based query processing
        response = "Sample AI-generated response based on legal documents."
        st.write(response)
    else:
        st.warning("⚠️ Please enter a question.")
