import streamlit as st
from streamlit_chat import message
from langchain.chat_models import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
import pinecone
import os
from dotenv import load_dotenv
from utils import *

# Load environment variables
load_dotenv()

# Hugging Face API Key
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")  # Ensure correct Pinecone environment

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index_name = "legal-articles"  # Ensure you have created this Pinecone index

# Initialize Hugging Face Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load Pinecone Vector Store
vector_store = Pinecone.from_existing_index(index_name, embedding_model)

# Retrieval-based QA using Pinecone
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})

# Hugging Face Model
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B",
    model_kwargs={"temperature": 0.5, "max_length": 512},
    huggingfacehub_api_token=HUGGINGFACE_API_KEY
)

# Prompt Template (Ensures direct retrieval, not AI-generated answers)
prompt_template = PromptTemplate(
    template="Extract and return only the exact text for {query}. If not found, say 'Article not available'.",
    input_variables=["query"]
)

# Retrieval Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_template}
)

# Streamlit UI
st.subheader("⚖️ AI-Powered Legal Chatbot (Hugging Face + Pinecone)")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["Hello! Enter the article number you want to retrieve."]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

# Chat UI Containers
response_container = st.container()
textcontainer = st.container()

with textcontainer:
    query = st.text_input("Enter article number (e.g., Article 22):", key="input")
    if query:
        with st.spinner("Retrieving..."):
            response = qa_chain.run(query)

        # Store conversation in session state
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)

with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')
