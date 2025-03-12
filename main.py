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

# 🔹 Load embedding model
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 🔹 Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    """Extracts text from a given PDF file."""
    reader = PdfReader(pdf_file)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

# 🔹 Function to store vectors in Pinecone
def store_vectors(embeddings, text_chunks, pdf_name):
    """Store chunked embeddings as articles in Pinecone."""
    upsert_data = []
    
    for idx, (embedding, text) in enumerate(zip(embeddings, text_chunks)):
        article_id = f"{pdf_name}_article_{idx}"  # Unique ID per chunk
        vector_id = f"{article_id}_{uuid.uuid4().hex[:8]}"  # Unique vector ID

        metadata = {
            "pdf_name": pdf_name,
            "article_id": article_id,
            "text": text
        }

        upsert_data.append((vector_id, embedding, metadata))

    if upsert_data:
        # 🔹 Store vectors in Pinecone
        index.upsert(vectors=upsert_data)

        # 🔹 Debug: Fetch back stored data to verify
        stored_data = index.query(vector=upsert_data[0][1], top_k=1, include_metadata=True)
        print("🔍 Sample stored metadata:", stored_data["matches"][0]["metadata"])

# 🔹 Streamlit UI
st.set_page_config(page_title="AI-Powered Legal HelpDesk", layout="wide")

st.header("📜 AI-Powered Legal HelpDesk")
st.subheader("Upload PDFs and Ask Questions")

# Sidebar: Display available PDFs
def get_stored_pdfs():
    """Fetch unique PDF names stored in Pinecone."""
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
        print(f"Error fetching PDFs: {e}")
        return []

# Upload & Process Multiple PDFs
uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        pdf_name = uploaded_file.name
        pdf_text = extract_text_from_pdf(uploaded_file)

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        text_chunks = text_splitter.split_text(pdf_text)

        # Generate embeddings
        embeddings = embed_model.embed_documents(text_chunks)

        # Store in Pinecone
        store_vectors(embeddings, text_chunks, pdf_name)

        st.success(f"✅ PDF '{pdf_name}' uploaded and processed successfully!")

# Refresh sidebar PDFs
stored_pdfs = get_stored_pdfs()
selected_pdf = st.sidebar.selectbox("Select a PDF", options=stored_pdfs if stored_pdfs else ["No PDFs Found"])

# Question Input
st.subheader("Ask a legal question:")
query = st.text_input("Type your question here...")

if query and selected_pdf != "No PDFs Found":
    # Search in Pinecone
    query_embedding = embed_model.embed_query(query)
    search_results = index.query(vector=query_embedding, top_k=5, include_metadata=True)

    st.subheader("📖 Relevant Legal Sections:")
    for match in search_results["matches"]:
        st.write(f"🔹 **From PDF:** {match['metadata']['pdf_name']}")
        st.write(f"📜 **Article ID:** {match['metadata']['article_id']}")
        st.write(match["metadata"].get("text", "No text available"))
        st.write("---")
