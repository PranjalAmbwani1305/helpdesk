import os
import pinecone
import streamlit as st
import PyPDF2
from dotenv import load_dotenv
from huggingface_hub import login
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 🔐 Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# ✅ Login to Hugging Face
login(token=HUGGINGFACE_TOKEN)

# ✅ Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY)
index_name = "helpdesk"
index = pinecone.Index(index_name)

# ✅ Load embedding model
embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en")

# ✅ Load Mixtral model for text generation
mixtral_model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(mixtral_model, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(
    mixtral_model, 
    torch_dtype=torch.float16,
    device_map="auto"
)

# 📄 Process PDF into chunks
def process_pdf(pdf_path, chunk_size=500):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# 📥 Store vectors in Pinecone
def store_vectors(chunks, pdf_name):
    for i, chunk in enumerate(chunks):
        vector = embedder.embed_documents([chunk])[0]
        index.upsert([(f"{pdf_name}-doc-{i}", vector, {"pdf_name": pdf_name, "text": chunk})])

# 🔍 Query vectors in Pinecone
def query_vectors(query):
    vector = embedder.embed_query(query)
    results = index.query(vector=vector, top_k=5, include_metadata=True)
    return results

# 🤖 Generate answer using Mixtral
def generate_answer(context, query):
    prompt = f"Legal Document Context:\n{context}\n\nUser's Question: {query}\n\nProvide a detailed legal response."
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 🎨 Streamlit UI
st.markdown("# 📜 AI-Powered Legal HelpDesk")

uploaded_file = st.file_uploader("📂 Upload a PDF", type=["pdf"])

if uploaded_file:
    temp_pdf_path = f"temp_{uploaded_file.name}"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    chunks = process_pdf(temp_pdf_path)
    store_vectors(chunks, uploaded_file.name)
    st.success("✅ PDF processed and indexed!")

query = st.text_input("💬 Ask a legal question:")

if st.button("Get Answer") and query:
    results = query_vectors(query)
    if results["matches"]:
        context_text = "\n\n".join([match["metadata"]["text"] for match in results["matches"]])
        answer = generate_answer(context_text, query)
        st.write("**📝 Answer:**", answer)
    else:
        st.warning("❌ No relevant information found.")
