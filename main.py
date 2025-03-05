import streamlit as st
import pinecone
import PyPDF2
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import re

# Load environment variables for Pinecone API key
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
index_name = "helpdesk"

# Initialize Pinecone
from pinecone import Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Try initializing the Pinecone index safely
try:
    index = pc.Index(index_name)
except Exception as e:
    st.error(f"Error initializing Pinecone index: {e}")
    index = None  # Prevent further execution if the index fails

# Initialize sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to process PDF, extract articles, and chunk them
def process_pdf(pdf_path, chunk_size=500):
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
        
        if not text.strip():
            return []  # If no text is extracted, return empty list

        sections = []
        current_title = None
        current_content = []
        article_pattern = r'^(Article \d+|Article [A-Za-z]+):.*$'
        paragraphs = text.split('\n')

        for para in paragraphs:
            if re.match(article_pattern, para.strip()):
                if current_title:
                    sections.append({'title': current_title, 'content': ' '.join(current_content)})
                current_title = para.strip()
                current_content = []
            else:
                current_content.append(para.strip())

        if current_title:
            sections.append({'title': current_title, 'content': ' '.join(current_content)})
        
        return sections
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return []

# Function to store vectors in Pinecone
def store_vectors(sections, pdf_name):
    if not index:
        st.error("Pinecone index not initialized.")
        return

    try:
        upserts = []
        for i, section in enumerate(sections):
            title = section['title']
            content = section['content']
            title_vector = model.encode(title).tolist()
            content_vector = model.encode(content).tolist()

            upserts.append((f"{pdf_name}-article-{i}-title", title_vector, {"pdf_name": pdf_name, "text": title, "type": "title"}))
            upserts.append((f"{pdf_name}-article-{i}-content", content_vector, {"pdf_name": pdf_name, "text": content, "type": "content"}))

        if upserts:
            index.upsert(upserts)
            st.success("PDF uploaded and processed successfully!")
    except Exception as e:
        st.error(f"Error storing vectors in Pinecone: {e}")

# Function to query vectors from Pinecone
def query_vectors(query, selected_pdf):
    if not index:
        return "Pinecone index not initialized."
    
    try:
        vector = model.encode(query).tolist()
        filter_condition = {"pdf_name": {"$eq": selected_pdf}} if selected_pdf else {}
        results = index.query(vector=vector, top_k=5, include_metadata=True, filter=filter_condition)

        if results and "matches" in results:
            matched_titles = []
            matched_contents = []

            for match in results["matches"]:
                if "metadata" in match:
                    if match["metadata"].get("type") == "title":
                        matched_titles.append(match["metadata"].get("text", ""))
                    elif match["metadata"].get("type") == "content":
                        matched_contents.append(match["metadata"].get("text", ""))

            return "\n\n".join(matched_titles) + "\n\n" + "\n\n".join(matched_contents) if matched_titles or matched_contents else "No relevant information found."
        else:
            return "No relevant information found."
    except Exception as e:
        return f"Error during query: {e}"

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>AI-Powered Legal HelpDesk</h1>", unsafe_allow_html=True)

pdf_source = st.radio("Select PDF Source", ("Upload from PC", "Choose from Document Storage"))

if pdf_source == "Upload from PC":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        temp_pdf_path = f"temp_{uploaded_file.name}"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())  # Safer file handling
        
        sections = process_pdf(temp_pdf_path)
        if sections:
            store_vectors(sections, uploaded_file.name)
        else:
            st.warning("No extractable text found in the PDF.")
else:
    st.info("Document Storage feature is currently unavailable.")

input_language = st.selectbox("Choose Input Language", ("English", "Arabic"))
response_language = st.selectbox("Choose Response Language", ("English", "Arabic"))

query = st.text_input("Ask a legal question:")

if st.button("Get Answer"):
    if query and uploaded_file:
        response = query_vectors(query, uploaded_file.name)
        st.write(f"**Answer:** {response}")
    elif not uploaded_file:
        st.warning("Please upload a PDF before querying.")
    else:
        st.warning("Please enter a legal question.")
