import streamlit as st
import pymongo
import pinecone
import os
import torch
from dotenv import load_dotenv
from deep_translator import GoogleTranslator  
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Load environment variables
load_dotenv()

# MongoDB Setup
MONGO_URI = os.getenv("MONGO_URI")
client = pymongo.MongoClient(MONGO_URI)
db = client["helpdesk"]
collection = db["data"]

# Pinecone Setup
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
from pinecone import Pinecone

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "helpdesk"

if index_name not in pc.list_indexes().names():
    pc.create_index(name=index_name, dimension=384, metric="cosine")  

index = pc.Index(index_name)

# Load Hugging Face embedding model
hf_model = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(hf_model)
embedder = AutoModelForCausalLM.from_pretrained(hf_model)

# Load Mixtral-8x22B-Instruct model
model_name = "mistralai/Mixtral-8x22B-Instruct-v0.1"
chat_tokenizer = AutoTokenizer.from_pretrained(model_name)
chat_model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# Function to get embeddings
def get_embedding(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        embedding = embedder(**tokens).last_hidden_state.mean(dim=1).squeeze().tolist()
    return embedding

# Function to generate AI response using Mixtral
def generate_ai_response(prompt):
    inputs = chat_tokenizer(prompt, return_tensors="pt").to(chat_model.device)
    with torch.no_grad():
        output = chat_model.generate(**inputs, max_length=500)
    return chat_tokenizer.decode(output[0], skip_special_tokens=True)

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>AI-Powered Legal HelpDesk</h1>", unsafe_allow_html=True)

query = st.text_input("Ask a legal question:")

if st.button("Get Answer"):
    if query:
        response = generate_ai_response(query)
        st.write(f"**Answer:** {response}")
    else:
        st.warning("Please enter a question.")
