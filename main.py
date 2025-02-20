import streamlit as st
import pinecone
from pinecone import Pinecone
import pymongo
import os
import pdfplumber
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
MONGO_URI = os.getenv("MONGO_URI")

# MongoDB Connection
client = pymongo.MongoClient(MONGO_URI)
db = client["helpdesk"]
collection = db["data"]

# Pinecone Initialization
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "helpdesk"
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

def store_pdf(pdf_name, pdf_data):
    """Stores uploaded PDFs in MongoDB."""
    if collection.count_documents({"pdf_name": pdf_name}) == 0:
        collection.insert_one({"pdf_name": pdf_name, "pdf_data": pdf_data})

def chunks_exist(pdf_name):
    """Checks if PDF chunks already exist in the database."""
    return collection.count_documents({"pdf_name": pdf_name}) > 0

def insert_chunks(chunks, pdf_name):
    """Inserts extracted text chunks into MongoDB."""
    if not chunks_exist(pdf_name):
        for chunk in chunks:
            collection.insert_one({"pdf_name": pdf_name, "text": chunk})

def store_vectors(chunks, pdf_name):
    """Embeds and stores document chunks in Pinecone."""
    vectors = embedding_model.encode(chunks).tolist()
    
    upserts = [(f"{pdf_name}-doc-{i}", vectors[i], {"pdf_name": pdf_name, "text": chunks[i]}) for i in range(len(chunks))]
    index.upsert(upserts)

def list_stored_pdfs():
    """Lists stored PDFs in MongoDB."""
    return collection.distinct("pdf_name")

def query_vectors(query, selected_pdf):
    """Searches Pinecone for the most relevant text chunks and formats the response."""
    query_vector = embedding_model.encode([query]).tolist()[0]
    
    results = index.query(vector=query_vector, top_k=8, include_metadata=True, filter={"pdf_name": {"$eq": selected_pdf}})

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

# Sidebar with stored PDFs
st.sidebar.header("📂 Stored PDFs")
pdf_list = list_stored_pdfs()
if pdf_list:
    with st.sidebar.expander("📜 View Stored PDFs", expanded=False):
        for pdf in pdf_list:
            st.sidebar.write(f"📄 {pdf}")
else:
    st.sidebar.write("No PDFs stored yet. Upload one!")

selected_pdf = None

# Select PDF source
pdf_source = st.radio("Select PDF Source", ["Upload from PC", "Choose from the Document Storage"])

if pdf_source == "Upload from PC":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        temp_pdf_path = f"temp_{uploaded_file.name}"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        store_pdf(uploaded_file.name, uploaded_file.read())

        if not chunks_exist(uploaded_file.name):
            chunks = process_pdf(temp_pdf_path)
            insert_chunks(chunks, uploaded_file.name)
            store_vectors(chunks, uploaded_file.name)
            st.success("✅ PDF uploaded and processed successfully!")
        else:
            st.info("ℹ️ This PDF has already been processed!")

        selected_pdf = uploaded_file.name

elif pdf_source == "Choose from the Document Storage":
    if pdf_list:
        selected_pdf = st.selectbox("📑 Select a PDF", pdf_list)
    else:
        st.warning("⚠️ No PDFs available in the repository. Please upload one.")

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
        
        response = query_vectors(detected_lang, selected_pdf)

        # Translate response to selected output language
        if response_lang == "Arabic":
            response = translate_text(response, "ar")
            st.markdown(f"<div dir='rtl' style='text-align: right;'>{response}</div>", unsafe_allow_html=True)
        else:
            st.write(f"**✅ Answer:** {response}")
    else:
        st.warning("⚠️ Please enter a query and select a PDF.")
