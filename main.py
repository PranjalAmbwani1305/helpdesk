import streamlit as st
import pinecone
import os
import re
from dotenv import load_dotenv
from deep_translator import GoogleTranslator  
from sentence_transformers import SentenceTransformer
import PyPDF2

# Load environment variables
load_dotenv()

# Initialize Hugging Face Model for Embeddings
hf_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Pinecone Connection
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
from pinecone import Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "helpdesk"
index = pc.Index(index_name)

# Function to Extract Text and Chunk by Article
def process_pdf(pdf_path, pdf_name, chunk_size=500):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
    # **Split into articles using "Article X" pattern**
    article_splits = re.split(r"(Article \d+)", text)
    chunks = []
    
    for i in range(1, len(article_splits), 2):
        title = article_splits[i]
        content = article_splits[i+1] if i+1 < len(article_splits) else ""
        
        # Break large articles into smaller chunks
        if len(content) > chunk_size:
            sub_chunks = [content[j:j+chunk_size] for j in range(0, len(content), chunk_size)]
            for part in sub_chunks:
                chunks.append((title, part))
        else:
            chunks.append((title, content))

    return chunks

# Check if a PDF is Already Stored in Pinecone
def pdf_already_stored(pdf_name):
    query_results = index.query(vector=[0]*384, top_k=1, filter={"pdf_name": {"$eq": pdf_name}})
    return bool(query_results["matches"])

# Store Vectors in Pinecone with Metadata
def store_vectors(chunks, pdf_name, chapter="Unknown Chapter"):
    vectors = []
    
    for i, (title, text) in enumerate(chunks):
        embedding = hf_model.encode(text).tolist()
        vector_id = f"{pdf_name}-{title.replace(' ', '-').lower()}"

        metadata = {
            "pdf_name": pdf_name,
            "text": text,
            "title": title,
            "chapter": chapter,
            "type": "article"
        }
        
        vectors.append((vector_id, embedding, metadata))
    
    if vectors:
        index.upsert(vectors)
        st.success(f"✅ {len(vectors)} articles stored successfully in Pinecone.")

# List Stored PDFs from Pinecone
def list_stored_pdfs():
    query_results = index.query(vector=[0]*384, top_k=100, include_metadata=True)
    pdf_names = set(match["metadata"]["pdf_name"] for match in query_results["matches"])
    return list(pdf_names)

# Query Pinecone for Relevant Information
def query_vectors(query, selected_pdf):
    vector = hf_model.encode(query).tolist()
    
    results = index.query(vector=vector, top_k=5, include_metadata=True, filter={"pdf_name": {"$eq": selected_pdf}})

    if results["matches"]:
        matched_texts = [match["metadata"]["text"] for match in results["matches"]]
        combined_text = "\n\n".join(matched_texts)
        return combined_text
    else:
        return "No relevant information found in the selected document."

# Translation Function
def translate_text(text, target_language):
    return GoogleTranslator(source="auto", target=target_language).translate(text)

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>AI-Powered Legal HelpDesk for Saudi Arabia</h1>", unsafe_allow_html=True)

st.sidebar.header("📂 Stored PDFs")
pdf_list = list_stored_pdfs()
if pdf_list:
    with st.sidebar.expander("📜 View Stored PDFs", expanded=False):
        for pdf in pdf_list:
            st.sidebar.write(f"📄 {pdf}")
else:
    st.sidebar.write("No PDFs stored yet. Upload one!")

selected_pdf = None

# PDF Selection
pdf_source = st.radio("Select PDF Source", ["Upload from PC", "Choose from the Document Storage"])

if pdf_source == "Upload from PC":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        temp_pdf_path = f"temp_{uploaded_file.name}"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        if not pdf_already_stored(uploaded_file.name):
            chunks = process_pdf(temp_pdf_path, uploaded_file.name)
            store_vectors(chunks, uploaded_file.name)  # Store with metadata
            st.success("PDF uploaded and processed!")
        else:
            st.info("This PDF has already been processed!")

        selected_pdf = uploaded_file.name

elif pdf_source == "Choose from the Document Storage":
    if pdf_list:
        selected_pdf = st.selectbox("Select a PDF", pdf_list)
    else:
        st.warning("No PDFs available in the repository. Please upload one!")

# Search Query
query = st.text_input("Ask a question related to the selected document:")

if query and selected_pdf:
    response = query_vectors(query, selected_pdf)
    st.write(response)

# Translation
if query:
    lang = st.selectbox("Translate Response To:", ["English", "Arabic", "French", "Hindi"])
    translated_text = translate_text(response, lang.lower())
    st.write(f"**Translated:** {translated_text}")
