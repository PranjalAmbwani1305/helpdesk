import streamlit as st
from pinecone import Pinecone
import pdfplumber
import os
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index_name = "helpdesk"

if index_name not in pc.list_indexes():
    pc.create_index(name=index_name, dimension=384, metric="cosine")

index = pc.Index(index_name)

# Load Hugging Face embedding model
embedding_model = SentenceTransformer("BAAI/bge-m3")

# Simulated document storage (Replace with database or actual storage system)
document_storage = ["Legal_Agreement.pdf", "Land_Act_2024.pdf", "Cyber_Crime_Laws.pdf"]

def process_pdf(pdf_path, chunk_size=500, overlap=100):
    """Extracts and chunks text from a PDF using a sliding window."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    # Sliding window chunking
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])

    return chunks

def store_vectors(chunks, pdf_name):
    """Embeds and stores chunks in Pinecone."""
    vectors = embedding_model.encode(chunks).tolist()
    
    upserts = [(f"{pdf_name}-chunk-{i}", vectors[i], {"pdf_name": pdf_name, "text": chunks[i]}) for i in range(len(chunks))]
    index.upsert(upserts)

def query_vectors(query, selected_pdf):
    """Searches for relevant text using Pinecone."""
    query_vector = embedding_model.encode([query]).tolist()[0]

    # Improved Query: Filters by selected PDF & returns best matches
    results = index.query(
        vector=query_vector, 
        top_k=5, 
        include_metadata=True, 
        filter={"pdf_name": {"$eq": selected_pdf}}
    )

    if results and "matches" in results:
        matched_texts = [(match["metadata"]["text"], match["score"]) for match in results["matches"]]

        # Sort results by cosine similarity score
        matched_texts = sorted(matched_texts, key=lambda x: x[1], reverse=True)
        return "\n\n".join([f"🔹 {text}" for text, _ in matched_texts])
    
    return "No relevant information found."

def translate_text(text, target_language):
    """Translates text to the target language."""
    return GoogleTranslator(source="auto", target=target_language).translate(text)

# Streamlit UI
st.title("🧑‍⚖️ AI-Powered Legal HelpDesk Chatbot")

# Select PDF Source
pdf_source = st.radio("📂 Select PDF Source", ["Upload from PC", "Choose from Document Storage"])
selected_pdf = None

if pdf_source == "Upload from PC":
    uploaded_file = st.file_uploader("📄 Upload a PDF", type=["pdf"])
    if uploaded_file:
        temp_pdf_path = f"temp_{uploaded_file.name}"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        chunks = process_pdf(temp_pdf_path)
        store_vectors(chunks, uploaded_file.name)
        st.success(f"✅ {uploaded_file.name} uploaded and processed!")
        selected_pdf = uploaded_file.name

elif pdf_source == "Choose from Document Storage":
    selected_pdf = st.selectbox("📁 Choose a legal document:", document_storage)
    st.success(f"📜 Using stored document: {selected_pdf}")

# Chatbot Interface
st.subheader("💬 Ask Your Legal Questions")
query_lang = st.radio("🌐 Query Language", ["English", "Arabic"], index=0)
response_lang = st.radio("🌐 Response Language", ["English", "Arabic"], index=0)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("💡 Your Question:")

if st.button("🕵️ Get Answer"):
    if selected_pdf and query:
        with st.spinner("🔎 Searching..."):
            query_translated = translate_text(query, "en") if query_lang == "Arabic" else query
            response = query_vectors(query_translated, selected_pdf)

        if response_lang == "Arabic":
            response = translate_text(response, "ar")
            response_display = f"<div dir='rtl' style='text-align: right;'>{response}</div>"
        else:
            response_display = f"**🤖 AI Answer:** {response}"

        # Add to chat history
        st.session_state.chat_history.append(("👤 You:", query))
        st.session_state.chat_history.append(("🤖 AI:", response_display))

        # Display chat history
        for speaker, message in st.session_state.chat_history:
            st.markdown(f"**{speaker}** {message}", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please select a document and ask a question.")
