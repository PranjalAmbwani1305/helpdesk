import streamlit as st
import pinecone
import PyPDF2
import os
from deep_translator import GoogleTranslator  
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load secrets from Streamlit
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENV"]

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index_name = "helpdesk"

index = pc.Index(index_name)

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Smaller, optimized Hugging Face model
generator = pipeline("text2text-generation", model="google/flan-t5-base")

def process_pdf(pdf_path, chunk_size=500):  # Reduced chunk size to avoid exceeding token limits
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def store_vectors(chunks, pdf_name):
    for i, chunk in enumerate(chunks):
        vector = embedder.encode(chunk).tolist()
        index.upsert([(f"{pdf_name}-doc-{i}", vector, {"pdf_name": pdf_name, "text": chunk})])

def query_vectors(query, selected_pdf):
    vector = embedder.encode(query).tolist()
    results = index.query(vector=vector, top_k=5, include_metadata=True, filter={"pdf_name": {"$eq": selected_pdf}})
    
    if results and "matches" in results and results["matches"]:
        matched_texts = [match["metadata"]["text"] for match in results["matches"]]
        combined_text = "\n\n".join(matched_texts)[:512]  # Ensure text does not exceed 512 tokens
        
        prompt = (
            f"You are an AI legal assistant. Based on the following extracted text from the document '{selected_pdf}', provide an accurate and well-formatted response with complete sentences and proper structure.\n\n"
            f"Document Excerpts:\n{combined_text}\n\n"
            f"User's Question: {query}\n\nAnswer: "
        )
        
        try:
            response = generator(prompt, max_length=512, truncation=True)[0]["generated_text"].strip()
        except Exception as e:
            response = f"Error generating response: {str(e)}"

        return response
    else:
        return "No relevant information found."

def translate_text(text, target_language):
    return GoogleTranslator(source="auto", target=target_language).translate(text)

st.markdown("<h1 style='text-align: center;'>AI-Powered Legal HelpDesk</h1>", unsafe_allow_html=True)

pdf_source = st.radio("Select PDF Source", ["Upload from PC"])
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

query = st.text_input("Ask a question:")

if st.button("Get Answer"):
    if uploaded_file and query:
        detected_lang = GoogleTranslator(source="auto", target="en").translate(query)
        response = query_vectors(detected_lang, uploaded_file.name)
        
        if response_lang == "Arabic":
            response = translate_text(response, "ar")
            st.markdown(f"<div dir='rtl' style='text-align: right; white-space: pre-wrap;'>{response}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p style='white-space: pre-wrap;'>{response}</p>", unsafe_allow_html=True)
    else:
        st.warning("Please enter a query and upload a PDF.")
