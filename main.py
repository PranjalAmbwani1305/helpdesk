import os
import streamlit as st
import pinecone
import PyPDF2
import numpy as np
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer

# --- Load environment variables ---
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

# --- Initialize Pinecone ---
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index_name = "pdf-qna"

if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, dimension=384, metric="cosine")  # Adjust based on model

index = pinecone.Index(index_name)

# --- Load Embedding Model ---
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# --- Function to Extract Text from PDF ---
def extract_text_from_pdf(pdf_path, chunk_size=500):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])

    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# --- Store Vectors in Pinecone ---
def store_vectors(chunks, pdf_name):
    for i, chunk in enumerate(chunks):
        vector = embedding_model.encode(chunk).tolist()
        index.upsert([(f"{pdf_name}-doc-{i}", vector, {"pdf_name": pdf_name, "text": chunk})])

# --- Retrieve Relevant Chunks ---
def query_vectors(query, selected_pdf):
    query_vector = embedding_model.encode(query).tolist()
    
    results = index.query(vector=query_vector, top_k=5, include_metadata=True, filter={"pdf_name": {"$eq": selected_pdf}})

    if results["matches"]:
        matched_texts = [match["metadata"]["text"] for match in results["matches"]]
        return "\n\n".join(matched_texts)
    else:
        return "No relevant information found in the selected document."

# --- Translate Text ---
def translate_text(text, target_language):
    return GoogleTranslator(source="auto", target=target_language).translate(text)

# --- Streamlit UI ---
st.markdown("<h1 style='text-align: center;'>AI-Powered Legal HelpDesk</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a Legal PDF", type=["pdf"])

if uploaded_file:
    temp_pdf_path = f"temp_{uploaded_file.name}"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    chunks = extract_text_from_pdf(temp_pdf_path)
    store_vectors(chunks, uploaded_file.name)
    st.success("PDF uploaded and processed!")

query = st.text_input("Ask a question (in English or Arabic):")
if st.button("Get Answer"):
    if uploaded_file and query:
        detected_lang = GoogleTranslator(source="auto", target="en").translate(query)
        response = query_vectors(detected_lang, uploaded_file.name)

        st.write(f"**Answer:** {response}")
    else:
        st.warning("Please enter a query and upload a PDF.")
