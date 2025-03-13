import streamlit as st
import os
import pinecone
import fitz  # PyMuPDF for PDF processing
from sentence_transformers import SentenceTransformer

# Load Pinecone API Key from Environment Variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "helpdesk"

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# Check if index exists, else create it
if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    pc.create_index(INDEX_NAME, dimension=768, metric="cosine")

index = pc.Index(INDEX_NAME)

# Load Embedding Model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Streamlit UI
st.title("📖 AI-Powered Legal HelpDesk")
st.subheader("Upload PDFs & Search Legal Documents")

# PDF Upload
uploaded_files = st.file_uploader("📂 Upload multiple PDFs", type="pdf", accept_multiple_files=True)

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

# Function to split text into smaller chunks
def split_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Process PDFs
if uploaded_files:
    for pdf in uploaded_files:
        st.success(f"Processing: {pdf.name}")
        pdf_text = extract_text_from_pdf(pdf)
        text_chunks = split_text(pdf_text)
        
        # Generate embeddings and store in Pinecone
        for i, chunk in enumerate(text_chunks):
            vector = embedder.encode(chunk).tolist()
            doc_id = f"{pdf.name}-chunk-{i}"
            index.upsert([(doc_id, vector, {"pdf_name": pdf.name, "text": chunk})])

    st.success("📂 PDFs uploaded and processed successfully!")

# Query Section
query = st.text_input("🔍 Enter your legal query:")
if query:
    query_vector = embedder.encode(query).tolist()
    results = index.query(vector=query_vector, top_k=5, include_metadata=True)

    if results.get("matches"):
        st.subheader("📜 Search Results")
        for match in results["matches"]:
            st.markdown(f"**📄 PDF:** {match['metadata']['pdf_name']}")
            st.write(f"🔹 {match['metadata']['text']}")
            st.write(f"📝 **Score:** {match['score']:.4f}")
            st.write("---")
    else:
        st.warning("No relevant results found.")
