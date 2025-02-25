import streamlit as st
import pinecone
from pinecone import Pinecone
import PyPDF2
import os
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load environment variables
load_dotenv()

# Pinecone API Keys from .env file
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENV"]

# Initialize Pinecone instance
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check and create index if not exist
index_name = "helpdesk"
if index_name not in pc.list_indexes().names():
    pc.create_index(name=index_name, dimension=768, metric="cosine")

index = pc.Index(index_name)

# Initialize Sentence Transformer for Embeddings
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Initialize Hugging Face Model for Text Generation
chatbot = pipeline("text-generation", model="distilgpt2")

# Function to process PDFs into text chunks
def process_pdf(pdf_path, chunk_size=500):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

# Store PDF content in Pinecone
def store_vectors(chunks, pdf_name):
    for i, chunk in enumerate(chunks):
        vector = embedder.encode(chunk).tolist()
        index.upsert([(f"{pdf_name}-doc-{i}", vector, {"pdf_name": pdf_name, "text": chunk})])

# Query Pinecone for relevant legal information
def query_vectors(query, selected_pdf):
    vector = embedder.encode(query).tolist()
    
    results = index.query(vector=vector, top_k=5, include_metadata=True, filter={"pdf_name": {"$eq": selected_pdf}})

    if results["matches"]:
        matched_texts = [match["metadata"]["text"] for match in results["matches"]]

        combined_text = "\n\n".join(matched_texts)

        prompt = (
            f"Based on the following legal document ({selected_pdf}), provide an accurate and well-reasoned answer:\n\n"
            f"{combined_text}\n\n"
            f"User's Question: {query}"
        )

        chat_response = chatbot(prompt, max_length=500, truncation=True)[0]["generated_text"]

        return chat_response
    else:
        return "No relevant information found in the selected document."

# Translate text using Google Translator
def translate_text(text, target_language):
    return GoogleTranslator(source="auto", target=target_language).translate(text)

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>AI-Powered Legal HelpDesk for Saudi Arabia</h1>", unsafe_allow_html=True)

# PDF Upload
pdf_source = st.radio("Select PDF Source", ["Upload from PC"])
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

# Language Selection
input_lang = st.radio("Choose Input Language", ["English", "Arabic"], index=0)
response_lang = st.radio("Choose Response Language", ["English", "Arabic"], index=0)

# Query Input
if input_lang == "Arabic":
    query = st.text_input("اسأل سؤالاً (باللغة العربية أو الإنجليزية):", key="query_input")
    st.markdown(
        "<style>.stTextInput>div>div>input { direction: rtl; text-align: right; }</style>",
        unsafe_allow_html=True
    )
else:
    query = st.text_input("Ask a question (in English or Arabic):", key="query_input")

# Handle Answer
if st.button("Get Answer"):
    if selected_pdf and query:
        detected_lang = GoogleTranslator(source="auto", target="en").translate(query)
        
        response = query_vectors(detected_lang, selected_pdf)

        if response_lang == "Arabic":
            response = translate_text(response, "ar")
            st.markdown(f"<div dir='rtl' style='text-align: right;'>{response}</div>", unsafe_allow_html=True)
        else:
            st.write(f"**Answer:** {response}")
    else:
        st.warning("Please enter a query and select a PDF.")
