import streamlit as st
import pinecone
import PyPDF2
import os
from dotenv import load_dotenv
from deep_translator import GoogleTranslator  
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index_name = "helpdesk"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # For sentence-transformers embeddings
        metric="cosine"
    )

index = pc.Index(index_name)

# Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def process_pdf(pdf_path, chunk_size=500):
    """Extracts text from a PDF and splits it into chunks."""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def store_vectors(chunks, pdf_name):
    """Embeds and stores PDF chunks in Pinecone."""
    vectors = [(f"{pdf_name}-doc-{i}", embedding_model.encode(chunk).tolist(), {"pdf_name": pdf_name, "text": chunk}) for i, chunk in enumerate(chunks)]
    index.upsert(vectors)

def query_vectors(query, selected_pdf):
    """Finds relevant text chunks from Pinecone based on query."""
    query_embedding = embedding_model.encode(query).tolist()
    results = index.query(vector=query_embedding, top_k=5, include_metadata=True, filter={"pdf_name": {"$eq": selected_pdf}})
    
    if results["matches"]:
        return "\n\n".join([match["metadata"]["text"] for match in results["matches"]])
    return "No relevant information found."

def translate_text(text, target_language):
    """Translates text into the specified language."""
    return GoogleTranslator(source="auto", target=target_language).translate(text)

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>AI-Powered Legal HelpDesk</h1>", unsafe_allow_html=True)

pdf_source = st.radio("Select PDF Source", ["Upload from PC", "Manual Selection"])
selected_pdf = None

if pdf_source == "Upload from PC":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        temp_pdf_path = f"temp_{uploaded_file.name}"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.read())
        chunks = process_pdf(temp_pdf_path)
        store_vectors(chunks, uploaded_file.name)
        st.success("PDF uploaded and processed!")
        selected_pdf = uploaded_file.name

elif pdf_source == "Manual Selection":
    selected_pdf = st.text_input("Enter the PDF name")

query = st.text_input("Ask a legal question:")
if st.button("Get Answer"):
    if selected_pdf and query:
        response = query_vectors(query, selected_pdf)
        st.write(f"**Answer:** {response}")
    else:
        st.warning("Please enter a query and select a PDF.")
