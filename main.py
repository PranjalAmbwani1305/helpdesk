import streamlit as st
import pinecone
import os
from dotenv import load_dotenv
from deep_translator import GoogleTranslator  
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import PyPDF2

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

from pinecone import Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "helpdesk"
index = pc.Index(index_name)

hf_model = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(hf_model)
model = AutoModel.from_pretrained(hf_model)

qa_pipeline = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1")

def get_embedding(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        embeddings = model(**tokens).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().tolist()

def process_pdf(pdf_path, chunk_size=500):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def store_vectors(chunks, pdf_name):
    for i, chunk in enumerate(chunks):
        vector = get_embedding(chunk)
        index.upsert([(f"{pdf_name}-doc-{i}", vector, {"pdf_name": pdf_name, "text": chunk})])

def query_vectors(query, selected_pdf):
    vector = get_embedding(query)
    results = index.query(vector=vector, top_k=5, include_metadata=True, filter={"pdf_name": {"$eq": selected_pdf}})

    if results["matches"]:
        matched_texts = [match["metadata"]["text"] for match in results["matches"]]
        combined_text = "\n\n".join(matched_texts)

        prompt = f"{combined_text}\n\nAnswer the question: {query}"
        response = qa_pipeline(prompt, max_length=300)[0]['generated_text']
        return response
    else:
        return "No relevant information found in the selected document."

def translate_text(text, target_language):
    return GoogleTranslator(source="auto", target=target_language).translate(text)

st.markdown("<h1 style='text-align: center;'>AI-Powered Legal HelpDesk for Saudi Arabia</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    temp_pdf_path = f"temp_{uploaded_file.name}"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.read())
    
    chunks = process_pdf(temp_pdf_path)
    store_vectors(chunks, uploaded_file.name)
    st.success("PDF uploaded and processed!")

query = st.text_input("Ask a question (in English or Arabic):")
response_lang = st.radio("Choose Response Language", ["English", "Arabic"], index=0)

if st.button("Get Answer"):
    if uploaded_file and query:
        detected_lang = GoogleTranslator(source="auto", target="en").translate(query)
        response = query_vectors(detected_lang, uploaded_file.name)
        if response_lang == "Arabic":
            response = translate_text(response, "ar")
            st.markdown(f"<div dir='rtl' style='text-align: right;'>{response}</div>", unsafe_allow_html=True)
        else:
            st.write(f"**Answer:** {response}")
    else:
        st.warning("Please enter a query and upload a PDF.")
