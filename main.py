import os
import streamlit as st
import pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east1-gcp")

# Initialize Pinecone client
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

INDEX_NAME = "helpdesk"
DIMENSION = 384  # Ensure this matches your embedding model
METRIC = "cosine"

# Check if index exists, otherwise create one
if INDEX_NAME not in [index["name"] for index in pc.list_indexes()]:
    pc.create_index(name=INDEX_NAME, dimension=DIMENSION, metric=METRIC)

# Connect to the existing index
index = pc.Index(INDEX_NAME)

# Load Hugging Face Embeddings
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Function to process PDF and extract text
def process_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PdfReader(file)
        text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

# Function to store vectors in Pinecone
def store_vectors(text, pdf_name):
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]  # Chunking the text
    vectors = [embedder.embed([chunk])[0] for chunk in chunks]

    upsert_data = [(f"{pdf_name}-{i}", vector, {"pdf_name": pdf_name, "text": chunk})
                   for i, (chunk, vector) in enumerate(zip(chunks, vectors))]

    index.upsert(vectors=upsert_data)

# Function to query Pinecone
def query_vectors(query, selected_pdf):
    query_vector = embedder.embed([query])[0]
    results = index.query(vector=query_vector, top_k=5, include_metadata=True, filter={"pdf_name": {"$eq": selected_pdf}})

    if results["matches"]:
        matched_texts = [match["metadata"]["text"] for match in results["matches"]]
        return "\n\n".join(matched_texts)
    else:
        return "No relevant information found."

# List stored PDFs in Pinecone
def list_stored_pdfs():
    return list(set([match["metadata"]["pdf_name"] for match in index.fetch([INDEX_NAME])["matches"]]))

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>AI-Powered Legal HelpDesk</h1>", unsafe_allow_html=True)

pdf_source = st.radio("Select PDF Source", ["Upload from PC", "Choose from Stored PDFs"])

if pdf_source == "Upload from PC":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        temp_pdf_path = f"temp_{uploaded_file.name}"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        text = process_pdf(temp_pdf_path)
        store_vectors(text, uploaded_file.name)
        st.success("PDF uploaded and processed!")

        selected_pdf = uploaded_file.name
else:
    pdf_list = list_stored_pdfs()
    if pdf_list:
        selected_pdf = st.selectbox("Select a PDF", pdf_list)
    else:
        st.warning("No PDFs available. Please upload one.")

query = st.text_input("Ask a legal question:")
if st.button("Get Answer"):
    if selected_pdf and query:
        response = query_vectors(query, selected_pdf)
        st.write(f"**Answer:** {response}")
    else:
        st.warning("Please select a PDF and enter a question.")
