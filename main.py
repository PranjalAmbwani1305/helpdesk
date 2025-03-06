import streamlit as st
import pinecone
import re
from sentence_transformers import SentenceTransformer

# Initialize Pinecone
PINECONE_API_KEY = "your_api_key"
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index_name = "helpdesk"

# Ensure index exists
if index_name not in pc.list_indexes().names():
    pc.create_index(name=index_name, dimension=1536, metric="cosine")

index = pc.Index(index_name)

# Load sentence embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Streamlit UI
st.title("📖 AI-Powered Legal Helpdesk")

# Sidebar - PDF Selection
st.sidebar.header("📂 Stored PDFs")
def list_stored_pdfs():
    # Dummy function to simulate stored PDFs
    return ["legal_doc_1.pdf", "legal_doc_2.pdf"]

pdf_list = list_stored_pdfs()
if pdf_list:
    with st.sidebar.expander("📜 View Stored PDFs", expanded=False):
        for pdf in pdf_list:
            st.sidebar.write(f"📄 {pdf}")
else:
    st.sidebar.write("No PDFs stored yet. Upload one!")

selected_pdf = None
pdf_source = st.radio("Select PDF Source", ["Upload from PC", "Choose from the Document Storage"])

# Handle PDF upload
if pdf_source == "Upload from PC":
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file:
        selected_pdf = uploaded_file.name
elif pdf_source == "Choose from the Document Storage":
    selected_pdf = st.selectbox("Select a stored PDF", pdf_list)

# Language Selection
st.subheader("Choose Input Language")
input_language = st.radio("", ("English", "Arabic"), horizontal=True)

st.subheader("Choose Response Language")
response_language = st.radio("", ("English", "Arabic"), horizontal=True, key="response_lang")

# Question Input
st.subheader("Ask a question (in English or Arabic):")
user_query = st.text_input("")

# Function to query Pinecone
def query_vectors(query, selected_pdf):
    if not selected_pdf:
        return "Please select a PDF first."

    try:
        vector = model.encode(query).tolist()

        # Detect chapter-related queries
        chapter_match = re.search(r'chapter\s*(\d+|one|two|three|four|five)', query, re.IGNORECASE)

        # Construct the filter query
        filter_query = {
            "pdf_name": {"$eq": selected_pdf},
            "type": {"$eq": "chapter"}
        }

        if chapter_match:
            chapter_number = chapter_match.group(1)
            filter_query["chapter_number"] = {"$eq": chapter_number}

        # Search in Pinecone
        query_result = index.query(vector=vector, filter=filter_query, top_k=3, include_metadata=True)

        if query_result and "matches" in query_result:
            results = query_result["matches"]
            if results:
                response_texts = [res["metadata"]["text"] for res in results]
                return "\n\n".join(response_texts)

        return "Sorry, I could not find an answer to your question."
    
    except Exception as e:
        return f"Error: {str(e)}"

# Handle query submission
if user_query:
    response = query_vectors(user_query, selected_pdf)
    st.subheader("Answer:")
    st.write(response)
