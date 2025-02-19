import streamlit as st
import pinecone
import PyPDF2
import os
from dotenv import load_dotenv
from deep_translator import GoogleTranslator  

load_dotenv()

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pinecone.init(api_key=PINECONE_API_KEY, environment="us-west1-gcp")
index_name = "pdf-qna"

if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1536, metric="cosine")

index = pinecone.Index(index_name)

# PDF Processing
def process_pdf(pdf_path, chunk_size=500):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

# Store PDF Locally
def store_pdf_locally(uploaded_file):
    pdf_path = os.path.join("pdf_storage", uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())
    return pdf_path

# Retrieve Stored PDFs
def list_stored_pdfs():
    return os.listdir("pdf_storage") if os.path.exists("pdf_storage") else []

# Pinecone Querying
def query_vectors(query, selected_pdf):
    results = index.query(vector=query, top_k=5, include_metadata=True)
    
    if results["matches"]:
        matched_texts = [match["metadata"]["text"] for match in results["matches"]]
        return "\n\n".join(matched_texts)
    else:
        return "No relevant information found."

# Translation
def translate_text(text, target_language):
    return GoogleTranslator(source="auto", target=target_language).translate(text)

# Streamlit UI
st.title("📜 AI-Powered Legal HelpDesk")

pdf_source = st.radio("Select PDF Source", ["Upload from PC", "Choose from Stored PDFs"])
selected_pdf = None

if pdf_source == "Upload from PC":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        os.makedirs("pdf_storage", exist_ok=True)
        selected_pdf = store_pdf_locally(uploaded_file)
        st.success("PDF uploaded and stored!")

elif pdf_source == "Choose from Stored PDFs":
    pdf_list = list_stored_pdfs()
    if pdf_list:
        selected_pdf = st.selectbox("Select a PDF", pdf_list)
    else:
        st.warning("No PDFs available. Please upload one.")

query = st.text_input("Ask a question:")

if st.button("Get Answer"):
    if selected_pdf and query:
        response = query_vectors(query, selected_pdf)
        st.write(f"**Answer:** {response}")
    else:
        st.warning("Please enter a query and select a PDF.")
