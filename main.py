import streamlit as st
import pinecone
from pinecone import Pinecone
import os
import pdfplumber
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator

# Load environment variables
load_dotenv()

# Pinecone Setup
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index_name = "helpdesk"

# Check if index exists, otherwise create one
if index_name not in pc.list_indexes().names():
    pc.create_index(name=index_name, dimension=384, metric="cosine")  # Hugging Face uses 384 dimensions

index = pc.Index(index_name)

# Hugging Face Sentence Embedding Model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def process_pdf(pdf_path, chunk_size=500):
    """Extracts and chunks text from a PDF file."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def store_vectors(chunks, pdf_name):
    """Embeds and stores document chunks in Pinecone."""
    vectors = embedding_model.encode(chunks).tolist()
    
    upserts = [(f"{pdf_name}-doc-{i}", vectors[i], {"text": chunks[i]}) for i in range(len(chunks))]
    index.upsert(upserts)

def query_vectors(query):
    """Searches Pinecone for the most relevant text chunks and formats the response."""
    query_vector = embedding_model.encode([query]).tolist()[0]
    
    results = index.query(vector=query_vector, top_k=8, include_metadata=True)

    if results and "matches" in results:
        matched_texts = [match["metadata"]["text"] for match in results["matches"]]

        # Join text properly and remove unnecessary newlines
        formatted_response = "\n\n".join(matched_texts).strip()

        return formatted_response if formatted_response else "No relevant information found."
    
    return "No relevant information found."

def translate_text(text, target_language):
    """Translates text to the target language using Google Translator."""
    return GoogleTranslator(source="auto", target=target_language).translate(text)

# Streamlit UI
st.markdown(
    "<h1 style='text-align: center;'>⚖️ AI-Powered Legal HelpDesk</h1>",
    unsafe_allow_html=True
)

# Upload PDF
uploaded_file = st.file_uploader("📂 Upload a PDF", type=["pdf"])
selected_pdf = None

if uploaded_file:
    temp_pdf_path = f"temp_{uploaded_file.name}"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    # Extract text and store vectors
    chunks = process_pdf(temp_pdf_path)
    store_vectors(chunks, uploaded_file.name)
    st.success("✅ PDF uploaded and processed successfully!")

    selected_pdf = uploaded_file.name

# Language selection
input_lang = st.radio("🌍 Choose Input Language", ["English", "Arabic"], index=0)
response_lang = st.radio("🌍 Choose Response Language", ["English", "Arabic"], index=0)

# Adjust input field based on language
if input_lang == "Arabic":
    query = st.text_input("💬 اسأل سؤالاً (باللغة العربية أو الإنجليزية):", key="query_input")
    st.markdown("<style>.stTextInput>div>div>input { direction: rtl; text-align: right; }</style>", unsafe_allow_html=True)
else:
    query = st.text_input("💬 Ask a question (in English or Arabic):", key="query_input")

# Submit query
if st.button("🔍 Get Answer"):
    if selected_pdf and query:
        # Translate Arabic query to English before searching
        detected_lang = GoogleTranslator(source="auto", target="en").translate(query)
        
        response = query_vectors(detected_lang)

        # Translate response to selected output language
        if response_lang == "Arabic":
            response = translate_text(response, "ar")
            st.markdown(f"<div dir='rtl' style='text-align: right;'>{response}</div>", unsafe_allow_html=True)
        else:
            st.write(f"**✅ Answer:** {response}")
    else:
        st.warning("⚠️ Please enter a query and upload a PDF.")
