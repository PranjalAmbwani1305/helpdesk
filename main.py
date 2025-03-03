import streamlit as st
from streamlit_chat import message
from langchain.chat_models import HuggingFaceHub
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from utils import *
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Hugging Face API Key
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Initialize Hugging Face Chat Model
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B",
    model_kwargs={"temperature": 0.5, "max_length": 512},
    huggingfacehub_api_token=HUGGINGFACE_API_KEY
)

# Streamlit UI
st.subheader("⚖️ AI-Powered Legal Chatbot (Hugging Face + Pinecone)")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["Hello! How can I assist you with legal queries?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

# Prompt Template
system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context. 
If the answer is not in the context, say 'I don't know'.""")

human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

# Conversation Chain
conversation = ConversationChain(
    memory=st.session_state.buffer_memory,
    prompt=prompt_template,
    llm=llm,
    verbose=True
)

# Chat UI Containers
response_container = st.container()
textcontainer = st.container()

with textcontainer:
    query = st.text_input("Enter your legal query:", key="input")
    if query:
        with st.spinner("Thinking..."):
            conversation_string = get_conversation_string()
            refined_query = query_refiner(conversation_string, query)
            
            st.subheader("🔍 Refined Query:")
            st.write(refined_query)

            # Retrieve context from Pinecone
            context = find_match(refined_query)
            
            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")

        # Store conversation in session state
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)

with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')
