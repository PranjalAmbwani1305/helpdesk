import streamlit as st
import PyPDF2
import os
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from pinecone import Pinecone  # ✅ Using the correct Pinecone import

# ✅ Load environment variables
load_dotenv()

# ✅ API Keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# ✅ Initialize Pinecone (Using Correct Method)
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "helpdesk"

index = pc.Index(index_name)  # ✅ Corrected Pinecone Index Access

# ✅ Initialize Hugging Face Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Load Pinecone Vector Store
vector_store = pc.from_existing_index(index_name, embedding_model)

# ✅ Retrieval-based QA using Pinecone
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ✅ Hugging Face Model for Answer Generation
llm = ChatOpenAI(model_name="mistralai/Mistral-7B-Instruct-v0.1", openai_api_key=HUGGINGFACE_API_KEY)

# ✅ Prompt Template for Precise Legal Text Retrieval
prompt_template = PromptTemplate(
    template="Extract and return only the exact legal text for {query}. If not found, say 'No relevant article found'.",
    input_variables=["query"]
)

# ✅ Retrieval Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_template}
)

# ✅ Function to Process PDFs into Text Chunks
def process_pdf(pdf_path, chunk_size=500):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

# ✅ Store PDF Chunks in Pinecone
def store_vectors(chunks, pdf_name):
    for i, chunk in enumerate(chunks):
        vector = embedding_model.embed_documents([chunk])[0]
        index.upsert([(f"{pdf_name}-doc-{i}", vector, {"pdf_name": pdf_name, "text": chunk})])

# ✅ Streamlit UI
st.markdown("<h1 style='text-align: center;'>⚖️ AI-Powered Legal HelpDesk</h1>", unsafe_allow_html=True)

pdf_source = st.radio("Select PDF Source", ["Upload from PC"])
selected_pdf = None

if pdf_source == "Upload from PC":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        temp_pdf_path = f"temp_{uploaded_file.name}"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        chunks = process_pdf(temp_pdf_path)
        store_vectors(chunks, uploaded_file.name)
        st.success("✅ PDF uploaded and processed successfully!")
        selected_pdf = uploaded_file.name

# ✅ Language Selection
input_lang = st.radio("Choose Input Language", ["English", "Arabic"], index=0)
response_lang = st.radio("Choose Response Language", ["English", "Arabic"], index=0)

query = st.text_input("Ask a legal question (e.g., 'Chapter 5' or 'Article 22'):" if input_lang == "English" else "اطرح سؤالاً قانونياً (مثال: 'الفصل 5'): ")

if st.button("Get Answer"):
    if selected_pdf and query:
        # ✅ Translate query to English for processing
        detected_lang = GoogleTranslator(source="auto", target="en").translate(query)
        response = qa_chain.run(detected_lang)

        # ✅ Translate response if needed
        if response_lang == "Arabic":
            response = GoogleTranslator(source="en", target="ar").translate(response)
            st.markdown(f"<div dir='rtl' style='text-align: right; color: #444;'>{response}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='color: #444;'>{response}</div>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a legal question and upload a PDF.")
