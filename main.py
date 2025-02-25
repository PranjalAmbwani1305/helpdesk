import streamlit as st
import pinecone
import openai
import PyPDF2
import os
from dotenv import load_dotenv
from deep_translator import GoogleTranslator  

# Load environment variables
load_dotenv()

# Load API keys
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Initialize OpenAI client
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# API key check
if "OPENAI_API_KEY" in st.secrets:
    st.write("✅ OpenAI API Key Loaded")
else:
    st.write("❌ OpenAI API Key Not Found")

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index_name = "helpdesk"
index = pc.Index(index_name)

if "PINECONE_API_KEY" in st.secrets:
    st.write("✅ Pinecone API Key Loaded")
else:
    st.write("❌ Pinecone API Key Not Found")
    
# PDF processing function
def process_pdf(pdf_path, chunk_size=500):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

# Store embeddings in Pinecone
def store_vectors(chunks, pdf_name):
    for i, chunk in enumerate(chunks):
        if not chunk.strip():  # Skip empty chunks
            continue
        try:
            response = openai_client.embeddings.create(input=[chunk], model="text-embedding-ada-002")
            vector = response.data[0].embedding
            index.upsert([(f"{pdf_name}-doc-{i}", vector, {"pdf_name": pdf_name, "text": chunk})])
        except Exception as e:
            st.error(f"Embedding error: {e}")

# Query embeddings from Pinecone
def query_vectors(query, selected_pdf):
    try:
        response = openai_client.embeddings.create(input=[query], model="text-embedding-ada-002")
        vector = response.data[0].embedding

        results = index.query(vector=vector, top_k=5, include_metadata=True, filter={"pdf_name": {"$eq": selected_pdf}})
        
        if "matches" in results and results["matches"]:
            matched_texts = [match["metadata"]["text"] for match in results["matches"]]
            combined_text = "\n\n".join(matched_texts)

            prompt = f"Based on the following legal document ({selected_pdf}), provide an answer:\n\n{combined_text}\n\nUser's Question: {query}"

            completion = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an AI assistant specialized in legal analysis."},
                    {"role": "user", "content": prompt}
                ]
            )

            return completion.choices[0].message.content
        else:
            return "No relevant information found in the selected document."
    except Exception as e:
        return f"Error: {e}"

# Translation function
def translate_text(text, target_language):
    return GoogleTranslator(source="auto", target=target_language).translate(text)

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>AI-Powered Legal HelpDesk for Saudi Arabia</h1>", unsafe_allow_html=True)

pdf_source = st.radio("Select PDF Source", ["Upload from PC"])
selected_pdf = None

if pdf_source == "Upload from PC":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        temp_pdf_path = f"temp_{uploaded_file.name}"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        chunks = process_pdf(temp_pdf_path)

        # Store embeddings for the PDF
        store_vectors(chunks, uploaded_file.name)

        st.success("PDF uploaded and processed!")
        selected_pdf = uploaded_file.name

input_lang = st.radio("Choose Input Language", ["English", "Arabic"], index=0)
response_lang = st.radio("Choose Response Language", ["English", "Arabic"], index=0)

query = st.text_input("Ask a question (in English or Arabic):", key="query_input")

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
