import streamlit as st
import pinecone
import openai
import PyPDF2
import os
from dotenv import load_dotenv
from deep_translator import GoogleTranslator  

# Load environment variables
load_dotenv()

# Pinecone API setup
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

from pinecone import Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "pdf-qna"
index = pc.Index(index_name)

# OpenAI API setup
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def process_pdf(pdf_path, chunk_size=500):
    """Extracts text from a PDF and splits it into chunks."""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def store_vectors(chunks, pdf_name):
    """Stores vector embeddings of text chunks in Pinecone."""
    for i, chunk in enumerate(chunks):
        vector = openai_client.embeddings.create(input=[chunk], model="text-embedding-ada-002").data[0].embedding
        index.upsert([(f"{pdf_name}-doc-{i}", vector, {"pdf_name": pdf_name, "text": chunk})])

def query_vectors(query, selected_pdf):
    """Retrieves relevant information from Pinecone based on query."""
    vector = openai_client.embeddings.create(input=[query], model="text-embedding-ada-002").data[0].embedding
    
    results = index.query(vector=vector, top_k=5, include_metadata=True, filter={"pdf_name": {"$eq": selected_pdf}})
    
    if results["matches"]:
        matched_texts = [match["metadata"]["text"] for match in results["matches"]]
        combined_text = "\n\n".join(matched_texts)
        
        prompt = (
            f"Based on the following legal document ({selected_pdf}), provide an accurate and well-reasoned answer:\n\n"
            f"{combined_text}\n\n"
            f"User's Question: {query}"
        )
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI assistant specialized in legal analysis."},
                {"role": "user", "content": prompt}
            ]
        ).choices[0].message.content

        return response
    else:
        return "No relevant information found in the selected document."

def translate_text(text, target_language):
    """Translates text to the target language."""
    return GoogleTranslator(source="auto", target=target_language).translate(text)

# Streamlit UI Setup
st.markdown(
    "<h1 style='text-align: center;'>AI-Powered Legal HelpDesk for Saudi Arabia</h1>",
    unsafe_allow_html=True
)

pdf_source = st.radio("Select PDF Source", ["Upload from PC", "Choose from the Document Storage in sidebar"])

if pdf_source == "Upload from PC":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        temp_pdf_path = f"temp_{uploaded_file.name}"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        chunks = process_pdf(temp_pdf_path)
        store_vectors(chunks, uploaded_file.name)
        st.success("PDF uploaded and processed!")

selected_pdf = uploaded_file.name if uploaded_file else None

# Language Selection
input_lang = st.radio("Choose Input Language", ["English", "Arabic"], index=0)
response_lang = st.radio("Choose Response Language", ["English", "Arabic"], index=0)

# Query Input
if input_lang == "Arabic":
    query = st.text_input("اسأل سؤالاً (باللغة العربية أو الإنجليزية):", key="query_input")
    st.markdown(
        """
        <style>
        .stTextInput>div>div>input {
            direction: rtl;
            text-align: right;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    query = st.text_input("Ask a question (in English or Arabic):", key="query_input")

# Query Processing
if st.button("Get Answer"):
    if selected_pdf and query:
        detected_lang = GoogleTranslator(source="auto", target="en").translate(query)
        response = query_vectors(detected_lang, selected_pdf)
        
        if response_lang == "Arabic":
            response = translate_text(response, "ar")
            st.markdown(f"<div dir='rtl' style='text-align: right;'>{response}</div>", unsafe_allow_html=True)
        else:
            st.write(f"**Answer:** {response}")
    else:
        st.warning("Please enter a query and select a PDF.")
