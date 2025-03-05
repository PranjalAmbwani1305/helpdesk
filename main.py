import asyncio

# Fix for "RuntimeError: no running event loop"
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.run(asyncio.sleep(0))

import streamlit as st
import pinecone
import os
import time
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# === Initialize Pinecone === #
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets.get("PINECONE_ENV", "us-east-1")
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index_name = "helpdesk"

if index_name not in pc.list_indexes().names():
    print("⚠️ Index does not exist. Creating index...")
    pc.create_index(name=index_name, dimension=384, metric="cosine")

time.sleep(5)
index = Pinecone.from_existing_index(index_name, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
print("✅ Pinecone Index Ready:", pc.describe_index(index_name))

# === Streamlit UI === #
def main():
    st.set_page_config(page_title="Saudi Legal HelpDesk", page_icon="⚖️")
    st.title("🏛️ AI-Powered Legal HelpDesk for Saudi Arabia")

    # Choose action
    action = st.radio("Choose an action:", ["Use existing PDFs", "Upload a new PDF"])

    if action == "Upload a new PDF":
        uploaded_file = st.file_uploader("📂 Upload a PDF", type=["pdf"])
        if uploaded_file:
            file_path = os.path.join("document_storage", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"✅ Document '{uploaded_file.name}' saved successfully!")

            # Process and store PDF in Pinecone
            loader = PyPDFLoader(file_path)
            docs = loader.load_and_split()
            index.add_documents(docs)

    # Retrieve available PDFs from Pinecone
    existing_pdfs = pc.describe_index(index_name).get("namespaces", {}).keys()
    if existing_pdfs:
        selected_pdf = st.selectbox("📖 Select PDF for Query", list(existing_pdfs), index=0)
    else:
        selected_pdf = None
        st.warning("⚠️ No PDFs found in Pinecone. Please upload a PDF first.")

    # User query input
    query = st.text_input("🔎 Ask a legal question:")

    if st.button("🔍 Get Answer"):
        if not selected_pdf:
            st.warning("⚠️ Please select a PDF before asking a question.")
            return
        
        if not query.strip():
            st.warning("⚠️ Please enter a valid query.")
            return

        # Initialize LangChain QA system
        retriever = index.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=OpenAI(api_key=OPENAI_API_KEY),
            retriever=retriever
        )

        response = qa_chain.run(query)
        st.write(f"**Answer:** {response}")

if __name__ == "__main__":
    main()
