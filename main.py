import streamlit as st
import pinecone
import PyPDF2
import os
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
st.set_page_config(page_title="Legal HelpDesk", layout="wide")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "helpdesk"

# --- INITIALIZE PINECONE ---
if not PINECONE_API_KEY:
    st.error("⚠️ Pinecone API key is missing. Set it as an environment variable.")
    st.stop()

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
existing_indexes = pc.list_indexes()
if INDEX_NAME not in existing_indexes:
    st.error(f"⚠️ Pinecone index '{INDEX_NAME}' not found. Create it first.")
    st.stop()

index = pc.Index(INDEX_NAME)

# --- LOAD EMBEDDING MODEL ---
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- UI ELEMENTS ---
st.sidebar.header("📄 Upload Legal Document")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
st.sidebar.markdown("---")

st.sidebar.header("🔍 Query HelpDesk")
query = st.sidebar.text_input("Enter a legal query", placeholder="e.g., Land acquisition law in India")
query_button = st.sidebar.button("Search")

# --- MAIN LOG DISPLAY ---
st.title("📚 AI-Powered Legal HelpDesk")
log_area = st.empty()

# --- FUNCTIONS ---
def clear_old_vectors():
    """Deletes all vectors from Pinecone before inserting new ones."""
    try:
        index.delete(delete_all=True)
        log_area.text("✅ Old vectors cleared from Pinecone.")
    except pinecone.openapi_support.exceptions.NotFoundException:
        log_area.text("⚠️ Index is empty or not found.")
    except Exception as e:
        st.error(f"Error while deleting vectors: {e}")

def extract_text_from_pdf(pdf_file):
    """Extracts text from uploaded PDF."""
    text = ""
    reader = PyPDF2.PdfReader(pdf_file)
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

def store_vectors(pdf_name, text):
    """Encodes and stores the latest PDF in Pinecone."""
    clear_old_vectors()
    vector = model.encode(text).tolist()
    index.upsert([(f"{pdf_name}-content", vector, {"pdf_name": pdf_name, "text": text})])
    log_area.text(f"✅ Stored '{pdf_name}' successfully in Pinecone.")

def query_helpdesk(user_query):
    """Retrieves the most relevant legal information."""
    query_vector = model.encode(user_query).tolist()
    results = index.query(queries=[query_vector], top_k=3, include_metadata=True)
    return results['matches'] if results else []

# --- UPLOAD HANDLING ---
if uploaded_file:
    with st.spinner("🔄 Processing PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)
        store_vectors(uploaded_file.name, pdf_text)
        st.success(f"📂 '{uploaded_file.name}' stored in Pinecone.")

# --- QUERY HANDLING ---
if query_button and query:
    with st.spinner("🔎 Searching legal database..."):
        results = query_helpdesk(query)
        if results:
            for match in results:
                st.markdown(f"**📜 Relevant Info:** {match['metadata']['text'][:500]}...")
        else:
            st.warning("⚠️ No relevant legal information found.")

