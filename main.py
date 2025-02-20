import os
import logging
import streamlit as st
import pymongo
import pinecone
import torch
import PyPDF2
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel, pipeline
from deep_translator import GoogleTranslator  


load_dotenv()


logging.basicConfig(level=logging.INFO)


MONGO_URI = os.getenv("MONGO_URI")
client = pymongo.MongoClient(MONGO_URI)
db = client["helpdesk"]
collection = db["data"]

# Pinecone Setup
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index_name = "pdf-qna"

if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, dimension=768, metric="cosine")

index = pinecone.Index(index_name)

# Hugging Face Setup
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOKENIZER_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
model = AutoModel.from_pretrained(TOKENIZER_MODEL, torch_dtype=torch.float16)
text_gen_pipeline = pipeline("text-generation", model=TOKENIZER_MODEL, tokenizer=tokenizer)

def process_pdf(pdf_path, chunk_size=500):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def chunks_exist(pdf_name):
    return collection.count_documents({"pdf_name": pdf_name}) > 0

def insert_chunks(chunks, pdf_name):
    if not chunks_exist(pdf_name):
        for chunk in chunks:
            collection.insert_one({"pdf_name": pdf_name, "text": chunk})

def store_vectors(chunks, pdf_name):
    for i, chunk in enumerate(chunks):
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
        vector = model(**inputs).last_hidden_state.mean(dim=1).detach().numpy().tolist()[0]
        index.upsert([(f"{pdf_name}-doc-{i}", vector, {"pdf_name": pdf_name, "text": chunk})])

def query_vectors(query, selected_pdf):
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    vector = model(**inputs).last_hidden_state.mean(dim=1).detach().numpy().tolist()[0]
    
    results = index.query(vector=vector, top_k=5, include_metadata=True, filter={"pdf_name": {"$eq": selected_pdf}})
    
    if results["matches"]:
        matched_texts = [match["metadata"]["text"] for match in results["matches"]]
        combined_text = "\n\n".join(matched_texts)
        prompt = f"Based on the following legal document ({selected_pdf}), provide an accurate answer:\n\n{combined_text}\n\nUser's Question: {query}"
        response = text_gen_pipeline(prompt, max_length=500, do_sample=True)[0]["generated_text"]
        return response
    else:
        return "No relevant information found in the selected document."

# Streamlit UI
st.title("🔹 AI-Powered Legal HelpDesk using Hugging Face & Pinecone")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    temp_pdf_path = f"temp_{uploaded_file.name}"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.read())
    
    chunks = process_pdf(temp_pdf_path)
    insert_chunks(chunks, uploaded_file.name)
    store_vectors(chunks, uploaded_file.name)
    st.success("PDF uploaded and processed!")

query = st.text_input("Ask a legal question:")
if st.button("Get Answer"):
    if uploaded_file and query:
        response = query_vectors(query, uploaded_file.name)
        st.write(f"**Answer:** {response}")
