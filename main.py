import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import AnalyzeDocumentChain
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.agents import create_openai_functions_agent
from langchain.llms import OpenAI
import openai
import os
import pinecone

# Initialize OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Setup Pinecone
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-west1-gcp")
index_name = "law-database"  # Update with your actual Pinecone index name

# Load and process the uploaded PDF
def process_pdf(file):
    # Use PyPDFLoader to load PDF
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    
    # Split text into manageable chunks for vectorization
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(pages)

    return documents

# Initialize vector store (Pinecone)
def create_pinecone_vector_store(documents):
    embeddings = OpenAIEmbeddings()
    docsearch = Pinecone.from_documents(documents, embeddings, index_name=index_name)
    return docsearch

# Handle user question
def get_answer(query, docsearch):
    # Perform the search
    results = docsearch.similarity_search(query)
    
    # Load the QA chain
    qa_chain = load_qa_chain(OpenAI(), chain_type="stuff")
    response = qa_chain.run(input_documents=results, question=query)
    
    return response

def main():
    # Streamlit interface
    st.title("AI-Powered Legal HelpDesk")

    # Upload PDF
    uploaded_file = st.file_uploader("Upload your Legal Document (PDF)", type="pdf")
    
    if uploaded_file is not None:
        # Process PDF
        st.write("Processing PDF...")
        documents = process_pdf(uploaded_file)
        
        # Create Pinecone vector store
        docsearch = create_pinecone_vector_store(documents)
        
        st.write("PDF Processed. You can now ask your legal questions.")

        # User query
        user_question = st.text_input("Ask a legal question:")

        if user_question:
            # Get answer
            st.write("Fetching answer...")
            answer = get_answer(user_question, docsearch)
            st.write("Answer:", answer)

if __name__ == "__main__":
    main()
