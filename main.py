import streamlit as st
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
import pinecone
import os
from dotenv import load_dotenv
from utils import get_conversation_string, query_refiner, find_match

# Load environment variables
load_dotenv()

# API Keys
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")  # Ensure the correct Pinecone environment

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index_name = "legal-helpdesk"

# Ensure the index exists
if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, dimension=768, metric="cosine")
index = pinecone.Index(index_name)

# Initialize Hugging Face Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load Pinecone Vector Store
vector_store = Pinecone.from_existing_index(index_name, embedding_model)

# Retrieval-based QA using Pinecone
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})

# Hugging Face Model for Answer Generation
llm = ChatOpenAI(model_name="mistralai/Mistral-7B-Instruct-v0.1", openai_api_key=HUGGINGFACE_API_KEY)

# Prompt Template for Better Formatting
prompt_template = PromptTemplate(
    template="Extract and return only the exact legal text for {query}. If not found, say 'No relevant article found'.",
    input_variables=["query"]
)

# Retrieval Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_template}
)

# Streamlit UI
st.markdown("<h1 style='text-align: center; color: #2E3B55;'>⚖️ AI-Powered Legal Chatbot</h1>", unsafe_allow_html=True)

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["Hello! Enter the chapter/article number to retrieve the exact text."]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

# Chat UI Containers
response_container = st.container()
textcontainer = st.container()

with textcontainer:
    query = st.text_input("Enter chapter/article number (e.g., 'Chapter 5' or 'Article 22'):", key="input")
    if query:
        with st.spinner("Retrieving legal text..."):
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
