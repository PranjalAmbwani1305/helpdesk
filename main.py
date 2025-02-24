import streamlit as st
import os
import pymongo
from pymongo import MongoClient
import pinecone
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB Connection (Optional for Tracking PDFs)
MONGO_URI = st.secrets["mongo"]["MONGO_URI"]
client = MongoClient(MONGO_URI)
db = client["helpdesk"]
pdf_collection = db["pdf_repository"]

# Load environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
index_name = "helpdesk"

# Initialize Pinecone using the Pinecone class
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# Connect to the Pinecone index
index = pc.Index(index_name)

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model_name="gpt-4", temperature=0.2, openai_api_key=OPENAI_API_KEY)

# Memory for chat history
memory = ConversationBufferMemory(memory_key="chat_history")

# Function to Process PDF
def process_pdf(uploaded_file):
    loader = PyPDFLoader(uploaded_file)
    pages = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(pages)
    return chunks

# Store PDF Metadata in MongoDB (Optional)
def store_pdf_metadata(pdf_name):
    if pdf_collection.count_documents({"pdf_name": pdf_name}) == 0:
        pdf_collection.insert_one({"pdf_name": pdf_name})

# List Stored PDFs from Pinecone
def list_stored_pdfs():
    try:
        results = index.describe_index_stats()
        if "namespaces" in results:
            return list(results["namespaces"].keys())  # Return PDF names from Pinecone
        return []
    except Exception as e:
        st.error(f"Error fetching PDFs from Pinecone: {str(e)}")
        return []

# Streamlit UI
st.title("📜 AI-Powered Legal HelpDesk for Saudi Arabia")

# Sidebar: List Stored PDFs
st.sidebar.header("📂 Stored PDFs")
stored_pdfs = list_stored_pdfs()
selected_pdf = st.sidebar.selectbox("Select a PDF", stored_pdfs) if stored_pdfs else None

# PDF Upload
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    pdf_name = uploaded_file.name
    store_pdf_metadata(pdf_name)  # Optional tracking in MongoDB
    pdf_chunks = process_pdf(uploaded_file)

    # Store embeddings in Pinecone
    for i, chunk in enumerate(pdf_chunks):
        vector = embeddings.embed_query(chunk.page_content)
        index.upsert(vectors=[(f"{pdf_name}-chunk-{i}", vector, {"pdf_name": pdf_name, "text": chunk.page_content})])
    
    st.success("PDF uploaded and processed!")

# Query Input
query = st.text_input("Ask a question about the document:")
if query and selected_pdf:
    vector = embeddings.embed_query(query)
    
    results = index.query(vector=vector, top_k=5, include_metadata=True, filter={"pdf_name": {"$eq": selected_pdf}})

    if results["matches"]:
        matched_texts = [match["metadata"]["text"] for match in results["matches"]]
        combined_text = "\n\n".join(matched_texts)
        
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=None, memory=memory)
        response = qa_chain.run(combined_text)
        
        st.write("**Answer:**", response)
    else:
        st.warning("No relevant information found in the selected document.")
