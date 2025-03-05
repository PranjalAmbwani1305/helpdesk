import streamlit as st
import pinecone
import PyPDF2
import torch
import re
import os
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator

# Load Hugging Face Embedding Model
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load Environment Variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index_name = "helpdesk"

index = pc.Index(index_name)

# Function to extract text and chunk it based on Chapters and Articles
def process_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])

    chapter_pattern = r"(?i)(Chapter\s+\d+|CHAPTER\s+[IVXLCDM]+)"
    article_pattern = r"(?i)(Article\s+\d+|ARTICLE\s+\d+)"

    chunks = []
    current_chunk = []
    current_title = None

    lines = text.split("\n")

    for line in lines:
        line = line.strip()

        # If a new chapter is found, save the previous chunk and start a new one
        if re.match(chapter_pattern, line):
            if current_chunk:
                chunks.append({"title": current_title, "content": " ".join(current_chunk)})
            current_title = line
            current_chunk = []

        # If an article is found, save the previous chunk and start a new one
        elif re.match(article_pattern, line):
            if current_chunk:
                chunks.append({"title": current_title, "content": " ".join(current_chunk)})
            current_title = line
            current_chunk = []

        # Otherwise, continue adding content
        else:
            current_chunk.append(line)

    # Add the last chunk
    if current_chunk:
        chunks.append({"title": current_title, "content": " ".join(current_chunk)})

    return chunks

# Function to store chunks in Pinecone
def store_vectors(chunks, pdf_name):
    for i, chunk in enumerate(chunks):
        vector = embed_model.encode(chunk["content"]).tolist()
        index.upsert([(f"{pdf_name}-doc-{i}", vector, {"pdf_name": pdf_name, "title": chunk["title"], "text": chunk["content"]})])

# Function to query Pinecone
def query_vectors(query, selected_pdf):
    vector = embed_model.encode(query).tolist()
    results = index.query(vector=vector, top_k=5, include_metadata=True, filter={"pdf_name": {"$eq": selected_pdf}})

    if results["matches"]:
        matched_texts = [match["metadata"]["text"] for match in results["matches"]]

        combined_text = "\n\n".join(matched_texts)

        return combined_text
    else:
        return "No relevant information found in the selected document."

# Function to translate text
def translate_text(text, target_language):
    return GoogleTranslator(source="auto", target=target_language).translate(text)

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>AI-Powered Legal HelpDesk for Saudi Arabia</h1>", unsafe_allow_html=True)

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

input_lang = st.radio("Choose Input Language", ["English", "Arabic"], index=0)
response_lang = st.radio("Choose Response Language", ["English", "Arabic"], index=0)

if input_lang == "Arabic":
    query = st.text_input("اسأل سؤالاً (باللغة العربية أو الإنجليزية):", key="query_input")
    query_html = "<style>.stTextInput>div>div>input { direction: rtl; text-align: right; }</style>"
    st.markdown(query_html, unsafe_allow_html=True)
else:
    query = st.text_input("Ask a question (in English or Arabic):", key="query_input")

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
