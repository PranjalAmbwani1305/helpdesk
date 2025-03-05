import streamlit as st
import PyPDF2
import pinecone
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from deep_translator import GoogleTranslator
from transformers import pipeline

# Download NLTK data for sentence tokenization
nltk.download("punkt")

# Initialize Pinecone
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENV"]

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index_name = "helpdesk"
index = pc.Index(index_name)

# Hugging Face Embedding Model (No OpenAI)
embedding_model = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")

# Function to Process PDF into Smart Chunks
def process_pdf(pdf_path, chunk_size=500):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])

    sentences = sent_tokenize(text)  # Split into sentences
    chunks, current_chunk = [], ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence  # Add sentence if within limit
        else:
            chunks.append(current_chunk.strip())  # Save full chunk
            current_chunk = sentence  # Start a new chunk
    
    if current_chunk:
        chunks.append(current_chunk.strip())  # Add last chunk

    return chunks

# Store Vectors in Pinecone
def store_vectors(chunks, pdf_name):
    for i, chunk in enumerate(chunks):
        embeddings = embedding_model(chunk)

        # Ensure correct embedding format
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings[0])
        else:
            embeddings = np.array(embeddings)

        if embeddings.ndim > 1:
            embeddings = embeddings.mean(axis=0)

        vector = embeddings.tolist()

        index.upsert([(f"{pdf_name}-doc-{i}", vector, {"pdf_name": pdf_name, "text": chunk})])

# Query Pinecone for Answers
def query_vectors(query, selected_pdf):
    embeddings = embedding_model(query)

    # Ensure correct format
    if isinstance(embeddings, list):
        embeddings = np.array(embeddings[0])
    else:
        embeddings = np.array(embeddings)

    if embeddings.ndim > 1:
        embeddings = embeddings.mean(axis=0)

    vector = embeddings.tolist()

    results = index.query(vector=vector, top_k=5, include_metadata=True, filter={"pdf_name": {"$eq": selected_pdf}})
    
    if results.matches:
        matched_texts = [match.metadata["text"] for match in results.matches]

        # Format response correctly
        response_text = "\n\n".join(matched_texts)
        response_text = response_text.replace("Article ", "\n\n**Article ").strip()

        return response_text
    else:
        return "No relevant information found."

# Translate Text
def translate_text(text, target_language):
    return GoogleTranslator(source="auto", target=target_language).translate(text)

# Streamlit UI
st.markdown(
    "<h1 style='text-align: center;'>📜 AI-Powered Legal HelpDesk for Saudi Arabia</h1>",
    unsafe_allow_html=True
)

# Select PDF Source
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    temp_pdf_path = f"temp_{uploaded_file.name}"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    chunks = process_pdf(temp_pdf_path)
    store_vectors(chunks, uploaded_file.name)
    st.success("✅ PDF uploaded and processed successfully!")
    selected_pdf = uploaded_file.name

# Language Selection
input_lang = st.radio("Choose Input Language", ["English", "Arabic"], index=0)
response_lang = st.radio("Choose Response Language", ["English", "Arabic"], index=0)

if input_lang == "Arabic":
    query = st.text_input("📝 اسأل سؤالاً (باللغة العربية أو الإنجليزية):", key="query_input")
    st.markdown(
        "<style>.stTextInput>div>div>input { direction: rtl; text-align: right; }</style>",
        unsafe_allow_html=True
    )
else:
    query = st.text_input("📝 Ask a question (in English or Arabic):", key="query_input")

# Get Answer
if st.button("Get Answer"):
    if uploaded_file and query:
        detected_lang = GoogleTranslator(source="auto", target="en").translate(query)
        response = query_vectors(detected_lang, selected_pdf)

        if response_lang == "Arabic":
            response = translate_text(response, "ar")
            st.markdown(f"<div dir='rtl' style='text-align: right;'>{response}</div>", unsafe_allow_html=True)
        else:
            st.write(f"**Answer:** {response}")
    else:
        st.warning("⚠️ Please enter a query and upload a PDF.")
