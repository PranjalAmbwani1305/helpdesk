import streamlit as st
import os
import PyPDF2
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load Hugging Face model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Directory for storing PDFs
PDF_STORAGE_DIR = "pdf_repository"
if not os.path.exists(PDF_STORAGE_DIR):
    os.makedirs(PDF_STORAGE_DIR)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    return text

# Function to store PDF text embeddings
def store_pdf_embeddings(pdf_name, pdf_text):
    embeddings = model.encode(pdf_text.split("\n"), convert_to_tensor=False)
    storage_path = os.path.join(PDF_STORAGE_DIR, f"{pdf_name}.pkl")
    with open(storage_path, "wb") as f:
        pickle.dump({"text": pdf_text, "embeddings": embeddings}, f)

# Function to check if a PDF is already stored
def is_pdf_stored(pdf_name):
    return os.path.exists(os.path.join(PDF_STORAGE_DIR, f"{pdf_name}.pkl"))

# Function to load stored PDF embeddings
def load_pdf_embeddings(pdf_name):
    storage_path = os.path.join(PDF_STORAGE_DIR, f"{pdf_name}.pkl")
    with open(storage_path, "rb") as f:
        return pickle.load(f)

# UI Title
st.markdown("<h1 style='text-align: center;'>AI-Powered Legal HelpDesk</h1>", unsafe_allow_html=True)

# Select PDF Source
pdf_source = st.radio("Select PDF Source", ["Upload New PDF", "Use Stored PDF"])

selected_pdf = None

if pdf_source == "Upload New PDF":
    uploaded_file = st.file_uploader("Upload a legal document (PDF)", type=["pdf"])
    if uploaded_file:
        pdf_name = uploaded_file.name
        pdf_path = os.path.join(PDF_STORAGE_DIR, pdf_name)
        
        if not is_pdf_stored(pdf_name):
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.read())
            pdf_text = extract_text_from_pdf(pdf_path)
            store_pdf_embeddings(pdf_name, pdf_text)
            st.success("PDF uploaded, processed, and stored!")
        else:
            st.info("PDF is already stored.")
        selected_pdf = pdf_name

elif pdf_source == "Use Stored PDF":
    stored_pdfs = [f.replace(".pkl", "") for f in os.listdir(PDF_STORAGE_DIR) if f.endswith(".pkl")]
    if stored_pdfs:
        selected_pdf = st.selectbox("Choose a stored PDF", stored_pdfs)
    else:
        st.warning("No stored PDFs available.")

# User Query
query = st.text_input("Ask a question (in English or Arabic):")

# Process query
if st.button("Get Answer"):
    if selected_pdf and query:
        pdf_data = load_pdf_embeddings(selected_pdf)
        pdf_text = pdf_data["text"].split("\n")
        pdf_embeddings = pdf_data["embeddings"]
        
        # Get query embedding
        query_embedding = model.encode(query, convert_to_tensor=False)
        
        # Find the most relevant passage using cosine similarity
        similarity_scores = cosine_similarity([query_embedding], pdf_embeddings)[0]
        best_match_index = similarity_scores.argmax()
        
        # Retrieve best matching text snippet
        matched_text = pdf_text[best_match_index]
        
        st.write(f"**Answer:** {matched_text}")
    else:
        st.warning("Please enter a query and select a PDF.")
