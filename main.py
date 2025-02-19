import streamlit as st
import pinecone
import pdfplumber
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator

# Load secrets
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENV"]
PINECONE_INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]

if PINECONE_INDEX_NAME:
    index = pc.Index(PINECONE_INDEX_NAME)
else:
    raise ValueError("PINECONE_INDEX_NAME is not set in Streamlit secrets!")
    
# Load Hugging Face embedding model (1536 dimensions)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>AI-Powered Legal HelpDesk</h1>", unsafe_allow_html=True)

pdf_file = st.sidebar.file_uploader("Upload a legal document (PDF)", type=["pdf"])

def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text

if pdf_file:
    doc_text = extract_text_from_pdf(pdf_file)
    st.sidebar.success("PDF uploaded and processed successfully!")
else:
    doc_text = ""

# Language selection
input_lang = st.radio("Choose Input Language", ["English", "Arabic"], index=0)
response_lang = st.radio("Choose Response Language", ["English", "Arabic"], index=0)

if input_lang == "Arabic":
    query = st.text_input("اسأل سؤالاً (باللغة العربية أو الإنجليزية):", key="query_input")
    st.markdown("<style>.stTextInput>div>div>input { direction: rtl; text-align: right; }</style>", unsafe_allow_html=True)
else:
    query = st.text_input("Ask a question (in English or Arabic):", key="query_input")

if st.button("Get Answer"):
    if doc_text and query:
        # Translate query if needed
        translated_query = GoogleTranslator(source="auto", target="en").translate(query)

        # Generate embedding for the query
        query_embedding = model.encode(translated_query).tolist()

        # Retrieve relevant chunks
        results = index.query(query_embedding, top_k=5, include_metadata=True)

        if results["matches"]:
            matched_texts = [match["metadata"]["text"] for match in results["matches"]]
            response = "\n\n".join(matched_texts)

            # Translate response if needed
            if response_lang == "Arabic":
                response = GoogleTranslator(source="auto", target="ar").translate(response)
                st.markdown(f"<div dir='rtl' style='text-align: right;'>{response}</div>", unsafe_allow_html=True)
            else:
                st.write(f"**Answer:** {response}")
        else:
            st.warning("No relevant information found in the document.")
    else:
        st.warning("Please enter a query and upload a PDF.")
