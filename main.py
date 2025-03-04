import streamlit as st
import pinecone
import os
import PyPDF2
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index_name = "helpdesk"

# Check if index exists
if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, dimension=384, metric="cosine")

index = pinecone.Index(index_name)

# Load models
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
qa_pipeline = pipeline("text-generation", model="tiiuae/falcon-7b-instruct")

# Function to process PDF
def process_pdf(pdf_path, chunk_size=500):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

# Function to store vectors in Pinecone
def store_vectors(chunks, pdf_name):
    for i, chunk in enumerate(chunks):
        vector = embedder.encode(chunk).tolist()
        index.upsert([(f"{pdf_name}-doc-{i}", vector, {"pdf_name": pdf_name, "text": chunk})])

# Function to query vectors
def query_vectors(query, selected_pdf):
    vector = embedder.encode(query).tolist()
    results = index.query(vector=vector, top_k=5, include_metadata=True, filter={"pdf_name": {"$eq": selected_pdf}})
    
    if results["matches"]:
        matched_texts = [match["metadata"]["text"] for match in results["matches"]]
        combined_text = "\n\n".join(matched_texts)

        prompt = (
            f"Based on the following legal document ({selected_pdf}), provide an accurate answer:\n\n"
            f"{combined_text}\n\n"
            f"User's Question: {query}"
        )

        response = qa_pipeline(prompt, max_length=200)[0]["generated_text"]
        return response
    else:
        return "No relevant information found in the selected document."

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>AI-Powered Legal HelpDesk</h1>", unsafe_allow_html=True)

st.sidebar.header("📂 Upload Legal Documents")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    temp_pdf_path = f"temp_{uploaded_file.name}"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    chunks = process_pdf(temp_pdf_path)
    store_vectors(chunks, uploaded_file.name)
    st.sidebar.success("PDF uploaded and processed!")

pdf_list = [match["metadata"]["pdf_name"] for match in index.query([], top_k=100, include_metadata=True)["matches"]]
selected_pdf = st.sidebar.selectbox("Select a PDF", options=pdf_list) if pdf_list else None

query = st.text_input("Ask a question about the document:")

if st.button("Get Answer"):
    if selected_pdf and query:
        response = query_vectors(query, selected_pdf)
        st.subheader("📖 AI Answer:")
        st.write(response)
    else:
        st.warning("Please enter a query and select a PDF.")
