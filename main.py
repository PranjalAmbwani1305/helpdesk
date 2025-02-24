import streamlit as st
import os
import pymongo
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

# MongoDB Connection
MONGO_URI = os.getenv("MONGO_URI")
client = pymongo.MongoClient(MONGO_URI)
db = client["helpdesk"]
pdf_collection = db["pdf_repository"]

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model_name="gpt-4", temperature=0.2, openai_api_key=OPENAI_API_KEY)

# Memory for chat history
memory = ConversationBufferMemory(memory_key="chat_history")

# PDF Processing Function
def process_pdf(uploaded_file):
    loader = PyPDFLoader(uploaded_file)
    pages = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    )
    chunks = text_splitter.split_documents(pages)
    return chunks

# Store PDF Metadata
def store_pdf_metadata(pdf_name, pdf_data):
    if pdf_collection.count_documents({"pdf_name": pdf_name}) == 0:
        pdf_collection.insert_one({"pdf_name": pdf_name, "pdf_data": pdf_data})

# Streamlit UI
st.title("AI-Powered Legal HelpDesk for Saudi Arabia")

# Sidebar: List Stored PDFs
st.sidebar.header("📂 Stored PDFs")
stored_pdfs = pdf_collection.distinct("pdf_name")
selected_pdf = st.sidebar.selectbox("Select a PDF", stored_pdfs) if stored_pdfs else None

# PDF Upload
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    pdf_name = uploaded_file.name
    store_pdf_metadata(pdf_name, uploaded_file.read())
    pdf_chunks = process_pdf(uploaded_file)
    vectorstore = FAISS.from_documents(pdf_chunks, embeddings)
    st.success("PDF uploaded and processed!")
else:
    st.warning("Upload a PDF to continue.")

# Query Input
query = st.text_input("Ask a question about the document:")
if query and selected_pdf:
    vectorstore = FAISS.load_local("faiss_index", embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, memory=memory)
    response = qa_chain.run(query)
    st.write("**Answer:**", response)

# Save vector index
if uploaded_file:
    vectorstore.save_local("faiss_index")
