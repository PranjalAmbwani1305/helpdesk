import os
import uuid
import streamlit as st
import pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

# 🔹 Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "helpdesk"

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# 🔹 Load Hugging Face embedding model
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 🔹 Streamlit UI Setup
st.set_page_config(page_title="AI-Powered Legal HelpDesk", layout="wide")

st.header("📜 AI-Powered Legal HelpDesk")
st.subheader("Select PDF Source")

# 🔹 Function to fetch stored PDFs
def get_stored_pdfs():
    try:
        stats = index.describe_index_stats()
        if "namespaces" in stats and "" in stats["namespaces"]:
            vector_count = stats["namespaces"][""]["vector_count"]
            if vector_count == 0:
                return []
            
            # Fetch stored vectors
            results = index.query(vector=[0] * 384, top_k=vector_count, include_metadata=True)

            # Extract unique PDF names
            pdf_names = list(set(
                match["metadata"]["pdf_name"] for match in results["matches"] if "pdf_name" in match["metadata"]
            ))

            return pdf_names
        return []
    except Exception as e:
        st.error(f"Error fetching PDFs: {e}")
        return []

# 🔹 Extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

# 🔹 Store vectors in Pinecone (Chunks stored as Articles)
def store_vectors(embeddings, text_chunks, pdf_name):
    upsert_data = []
    for idx, (embedding, text) in enumerate(zip(embeddings, text_chunks)):
        article_id = f"{pdf_name}_article_{idx}"  # Unique Article ID
        vector_id = f"{article_id}_{uuid.uuid4().hex[:8]}"  # Unique vector ID

        upsert_data.append((vector_id, embedding, {"pdf_name": pdf_name, "article_id": article_id, "text": text}))

    if upsert_data:
        index.upsert(vectors=upsert_data)

# 🔹 Sidebar - Show stored PDFs
st.sidebar.header("📂 Stored PDFs")
stored_pdfs = get_stored_pdfs()
selected_pdf = st.sidebar.selectbox("Select a PDF", options=stored_pdfs if stored_pdfs else ["No PDFs Found"])

# 🔹 Upload & Process PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    pdf_name = uploaded_file.name
    pdf_text = extract_text_from_pdf(uploaded_file)

    # Split text into chunks (Articles)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_text(pdf_text)

    # Generate embeddings
    embeddings = embed_model.embed_documents(text_chunks)

    # Store in Pinecone
    store_vectors(embeddings, text_chunks, pdf_name)

    st.success(f"✅ PDF '{pdf_name}' uploaded and processed successfully!")

    # Refresh dropdown
    stored_pdfs = get_stored_pdfs()
    st.sidebar.selectbox("Select a PDF", options=stored_pdfs if stored_pdfs else ["No PDFs Found"], key="pdf_dropdown")

# 🔹 Question Input
st.subheader("Ask a legal question:")
query = st.text_input("Type your question here...")

if query and selected_pdf != "No PDFs Found":
    query_embedding = embed_model.embed_query(query)
    search_results = index.query(vector=query_embedding, top_k=5, include_metadata=True)

    st.subheader("📖 Relevant Legal Sections:")
    article_results = {}

    for match in search_results["matches"]:
        pdf_name = match["metadata"].get("pdf_name", "Unknown PDF")
        article_id = match["metadata"].get("article_id", f"unknown_{uuid.uuid4().hex[:8]}")
        text = match["metadata"].get("text", "No text available")

        if article_id not in article_results:
            article_results[article_id] = {"pdf_name": pdf_name, "text": []}
        
        article_results[article_id]["text"].append(text)

    # Display results
    for article_id, data in article_results.items():
        st.write(f"🔹 **From PDF:** {data['pdf_name']}")
        st.write("\n".join(data["text"]))
        st.write("---")
