import streamlit as st
import pymongo
import pinecone
import os
import torch
import PyPDF2
from dotenv import load_dotenv
from deep_translator import GoogleTranslator  
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModel

# Load environment variables
load_dotenv()

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI")
client = pymongo.MongoClient(MONGO_URI)
db = client["helpdesk"]
collection = db["data"]
pdf_collection = db["pdfs"]

# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index_name = "helpdesk"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # Adjusted for the embedding model
        metric="cosine"
    )

index = pc.Index(index_name)

# Hugging Face Authentication
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Load Hugging Face Embedding Model
hf_model = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(hf_model)
embedder = AutoModel.from_pretrained(hf_model)

# Load Hugging Face Chat Model (Mixtral-8x22B)
model_name = "mistralai/Mixtral-8x22B-Instruct-v0.1"
chat_tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
chat_model = AutoModelForCausalLM.from_pretrained(
    model_name, token=HF_TOKEN, 
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# Function to get text embeddings
def get_embedding(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        embedding = embedder(**tokens).last_hidden_state.mean(dim=1).squeeze().tolist()
    return embedding

# Function to process PDFs and extract text
def process_pdf(pdf_path, chunk_size=500):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

# Check if chunks exist for a given PDF
def chunks_exist(pdf_name):
    return collection.count_documents({"pdf_name": pdf_name}) > 0

# Insert extracted text chunks into MongoDB
def insert_chunks(chunks, pdf_name):
    if not chunks_exist(pdf_name):
        for chunk in chunks:
            collection.insert_one({"pdf_name": pdf_name, "text": chunk})

# Store vector embeddings in Pinecone
def store_vectors(chunks, pdf_name):
    for i, chunk in enumerate(chunks):
        vector = get_embedding(chunk)
        index.upsert([(f"{pdf_name}-doc-{i}", vector, {"pdf_name": pdf_name, "text": chunk})])

# List stored PDFs
def list_stored_pdfs():
    return pdf_collection.distinct("pdf_name")

# Store PDF metadata in MongoDB
def store_pdf(pdf_name, pdf_data):
    if pdf_collection.count_documents({"pdf_name": pdf_name}) == 0:
        pdf_collection.insert_one({"pdf_name": pdf_name, "pdf_data": pdf_data})

# Query the Pinecone index for similar text chunks
def query_vectors(query, selected_pdf):
    vector = get_embedding(query)
    
    results = index.query(vector=vector, top_k=5, include_metadata=True, filter={"pdf_name": {"$eq": selected_pdf}})
    
    if results["matches"]:
        matched_texts = [match["metadata"]["text"] for match in results["matches"]]
        combined_text = "\n\n".join(matched_texts)
        
        prompt = (
            f"Based on the following legal document ({selected_pdf}), provide an accurate and well-reasoned answer:\n\n"
            f"{combined_text}\n\n"
            f"User's Question: {query}"
        )
        
        inputs = chat_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        output = chat_model.generate(**inputs, max_length=500, do_sample=True, temperature=0.7)
        response = chat_tokenizer.decode(output[0], skip_special_tokens=True)
        
        return response
    else:
        return "No relevant information found in the selected document."

# Function to translate text
def translate_text(text, target_language):
    return GoogleTranslator(source="auto", target=target_language).translate(text)

# Streamlit UI
st.markdown(
    "<h1 style='text-align: center;'>AI-Powered Legal HelpDesk for Saudi Arabia</h1>",
    unsafe_allow_html=True
)

# Sidebar for stored PDFs
st.sidebar.header("📂 Stored PDFs")
pdf_list = list_stored_pdfs()
if pdf_list:
    with st.sidebar.expander("📜 View Stored PDFs", expanded=False):
        for pdf in pdf_list:
            st.sidebar.write(f"📄 {pdf}")
else:
    st.sidebar.write("No PDFs stored yet. Upload one!")

# PDF selection
selected_pdf = None
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
            st.success("PDF uploaded and processed!")
        else:
            st.info("This PDF has already been processed!")

        selected_pdf = uploaded_file.name

elif pdf_source == "Choose from the Document Storage":
    if pdf_list:
        selected_pdf = st.selectbox("Select a PDF", pdf_list)
    else:
        st.warning("No PDFs available in the repository. Please upload one.")

# User input for language and query
input_lang = st.radio("Choose Input Language", ["English", "Arabic"], index=0)
response_lang = st.radio("Choose Response Language", ["English", "Arabic"], index=0)

query = st.text_input("Ask a question (in English or Arabic):", key="query_input")

# Process query
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
        st.warning("Please enter a query and select a PDF.")
