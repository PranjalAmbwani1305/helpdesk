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

# Function to extract and chunk text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

# Function to extract article and chapter numbers
def extract_metadata(text):
    article_match = re.search(r"Article\s(\d+)", text, re.IGNORECASE)
    chapter_match = re.search(r"Chapter\s(\d+):\s([\w\s]+)", text, re.IGNORECASE)
    article_number = article_match.group(1) if article_match else "Unknown"
    chapter_number = chapter_match.group(1) if chapter_match else "Unknown"
    chapter_name = chapter_match.group(2) if chapter_match else "Unknown"
    return article_number, chapter_number, chapter_name

# Function to chunk text into smaller passages
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Function to store chunks in Pinecone
def store_chunks_in_pinecone(chunks, pdf_name):
    vectors = []
    for i, chunk in enumerate(chunks):
        article_number, chapter_number, chapter_name = extract_metadata(chunk)
        embedding = model.encode(chunk).tolist()
        vectors.append({
            "id": f"{pdf_name}-article-{article_number}",
            "values": embedding,
            "metadata": {
                "text": chunk,
                "article_number": article_number,
                "chapter_number": chapter_number,
                "chapter_name": chapter_name,
                "pdf_name": pdf_name,
                "type": "article"
            }
        })
    index.upsert(vectors=vectors)

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>AI-Powered Legal HelpDesk for Saudi Arabia</h1>", unsafe_allow_html=True)

# Sidebar for stored PDFs
st.sidebar.header("📂 Stored PDFs")
if pdf_list:
    with st.sidebar.expander("📝 View Stored PDFs", expanded=False):
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
        
        # Process and store in Pinecone
        extracted_text = extract_text_from_pdf(temp_pdf_path)
        text_chunks = chunk_text(extracted_text)
        store_chunks_in_pinecone(text_chunks, uploaded_file.name)
        st.success("PDF content has been processed and stored in Pinecone!")
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
