import streamlit as st
import pinecone
import os
from dotenv import load_dotenv
from deep_translator import GoogleTranslator  
from transformers import pipeline, AutoTokenizer, AutoModel
import PyPDF2

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

from pinecone import Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "helpdesk"

index = pc.Index(index_name)

model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
embedding_model = AutoModel.from_pretrained(model_name)
qa_pipeline = pipeline("text2text-generation", model="MBZUAI/LaMini-T5-738M")

def process_pdf(pdf_path, chunk_size=500):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = embedding_model(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()
    return embeddings

def store_vectors(chunks, pdf_name):
    for i, chunk in enumerate(chunks):
        vector = embed_text(chunk)
        index.upsert([(f"{pdf_name}-doc-{i}", vector, {"pdf_name": pdf_name, "text": chunk})])

def query_vectors(query, selected_pdf):
    vector = embed_text(query)
    results = index.query(vector=vector, top_k=5, include_metadata=True, filter={"pdf_name": {"$eq": selected_pdf}})
    
    if results["matches"]:
        matched_texts = [match["metadata"]["text"] for match in results["matches"]]
        combined_text = "\n\n".join(matched_texts)
        prompt = f"Based on the following legal document ({selected_pdf}), provide an accurate answer:\n\n{combined_text}\n\nUser's Question: {query}"
        response = qa_pipeline(prompt)[0]['generated_text']
        return response
    else:
        return "No relevant information found in the selected document."

def translate_text(text, target_language):
    return GoogleTranslator(source="auto", target=target_language).translate(text)

st.markdown("<h1 style='text-align: center;'>AI-Powered Legal HelpDesk for Saudi Arabia</h1>", unsafe_allow_html=True)

selected_pdf = None
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
    if selected_pdf and query:
        detected_lang = GoogleTranslator(source="auto", target="en").translate(query)
        response = query_vectors(detected_lang, selected_pdf)
        
        if response_lang == "Arabic":
            response = translate_text(response, "ar")
            st.markdown(f"<div dir='rtl' style='text-align: right;'>{response}</div>", unsafe_allow_html=True)
        else:
            st.write(f"**Answer:** {response}")
    else:
        st.warning("Please enter a query and upload a PDF.")
