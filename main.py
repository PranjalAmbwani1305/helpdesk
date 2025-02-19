import streamlit as st
import pinecone
import PyPDF2
import os
from dotenv import load_dotenv
from deep_translator import GoogleTranslator  
from sentence_transformers import SentenceTransformer  

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "helpdesk"

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

from pinecone import Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(INDEX_NAME)

def process_pdf(pdf_path, chunk_size=500):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def store_vectors(chunks, pdf_name):
    for i, chunk in enumerate(chunks):
        try:
            vector = model.encode(chunk).tolist()
            index.upsert([(f"{pdf_name}-doc-{i}", vector, {"pdf_name": pdf_name, "text": chunk})])
        except Exception as e:
            st.error(f"Vector storage failed: {str(e)}")

def query_vectors(query, selected_pdf):
    try:
        vector = model.encode(query).tolist()
        results = index.query(vector=vector, top_k=7, include_metadata=True, filter={"pdf_name": {"$eq": selected_pdf}})

        if results["matches"]:
            matched_texts = [match["metadata"]["text"] for match in results["matches"]]
            combined_text = "\n\n".join(matched_texts)

            return combined_text
        else:
            return "No relevant information found in the selected document."
    except Exception as e:
        return f"Error retrieving answer: {str(e)}"

def translate_text(text, target_language):
    return GoogleTranslator(source="auto", target=target_language).translate(text)

st.markdown(
    "<h1 style='text-align: center;'>AI-Powered Legal HelpDesk for Saudi Arabia</h1>",
    unsafe_allow_html=True
)

st.sidebar.header("📂 Select Document Source")
pdf_source = st.sidebar.radio("Choose Source", ["Upload a Document", "Pick from Stored Documents"])

uploaded_file = None
selected_pdf = None
pdf_list = ["Sample_Law_Document.pdf"]

if pdf_source == "Pick from Stored Documents":
    selected_pdf = st.sidebar.selectbox("Select a Document", pdf_list)

elif pdf_source == "Upload a Document":
    uploaded_file = st.sidebar.file_uploader("Drag & Drop or Browse to Add a Document", type=["pdf"])
    if uploaded_file:
        temp_pdf_path = f"temp_{uploaded_file.name}"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        selected_pdf = uploaded_file.name
        chunks = process_pdf(temp_pdf_path)
        store_vectors(chunks, selected_pdf)
        st.sidebar.success("Document uploaded and processed!")

input_lang = st.radio("Select Question Language", ["English", "Arabic"], index=0)
response_lang = st.radio("Select Response Language", ["English", "Arabic"], index=0)

if input_lang == "Arabic":
    query = st.text_input("اكتب سؤالك هنا (باللغة العربية أو الإنجليزية):", key="query_input")
    st.markdown(
        "<style>.stTextInput>div>div>input {direction: rtl; text-align: right;}</style>",
        unsafe_allow_html=True
    )
else:
    query = st.text_input("Enter Your Question (English or Arabic):", key="query_input")

if st.button("Get Legal Insight"):
    if selected_pdf and query:
        detected_lang = GoogleTranslator(source="auto", target="en").translate(query)
        response = query_vectors(detected_lang, selected_pdf)

        if response_lang == "Arabic":
            response = translate_text(response, "ar")
            st.markdown(f"<div dir='rtl' style='text-align: right;'>{response}</div>", unsafe_allow_html=True)
        else:
            st.write(f"**Legal Response:** {response}")
    else:
        st.warning("Please enter a question and select a document.")
