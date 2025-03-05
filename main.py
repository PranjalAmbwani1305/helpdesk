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

from pinecone import Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Initialize Pinecone index
index = pc.Index(index_name)

# Initialize sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to process PDF, extract titles and contents, and chunk them
def process_pdf(pdf_path, chunk_size=500):
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
        
        if not text:
            raise ValueError("No text extracted from PDF. Please check the PDF file.")
        
        sections = []
        current_title = None
        current_content = []

        # Refined regex to detect articles like "Article 1: Title" or "Article One: Title"
        article_pattern = r'^(Article \d+|Article [A-Za-z]+):.*$'

        # Split text into paragraphs or sections
        paragraphs = text.split('\n')

        for para in paragraphs:
            # Check if the paragraph looks like an article title
            if re.match(article_pattern, para.strip()):
                if current_title:
                    sections.append({'title': current_title, 'content': ' '.join(current_content)})
                current_title = para.strip()  # Set new title
                current_content = []  # Reset content
            else:
                current_content.append(para.strip())
        
        # Add the last section
        if current_title:
            sections.append({'title': current_title, 'content': ' '.join(current_content)})
        
        return sections

    except Exception as e:
        st.error(f"Error processing the PDF: {e}")
        return []

# Function to store vectors of title and content (chunked by article)
def store_vectors(sections, pdf_name):
    for i, section in enumerate(sections):
        title = section['title']
        content = section['content']
        
        # Create a vector for both title and content
        title_vector = model.encode(title).tolist()
        content_vector = model.encode(content).tolist()
        
        # Store title and content vectors in Pinecone
        index.upsert([(
            f"{pdf_name}-article-{i}-title", title_vector, {"pdf_name": pdf_name, "text": title, "type": "title"}),
            (f"{pdf_name}-article-{i}-content", content_vector, {"pdf_name": pdf_name, "text": content, "type": "content"})
        ])

# Function to query vectors from Pinecone
def query_vectors(query, selected_pdf):
    try:
        vector = model.encode(query).tolist()
        results = index.query(vector=vector, top_k=5, include_metadata=True, filter={"pdf_name": {"$eq": selected_pdf}})
        
        if results["matches"]:
            matched_texts = [match["metadata"]["text"] for match in results["matches"]]
            return "\n\n".join(matched_texts)
        else:
            return "No relevant information found in the selected document."
    except Exception as e:
        return f"Error during query: {e}"

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>AI-Powered Legal HelpDesk</h1>", unsafe_allow_html=True)

# PDF Source Selection: Upload or Choose from Document Storage
pdf_source = st.radio("Select PDF Source", ("Upload from PC", "Choose from Document Storage"))

uploaded_file = None
if pdf_source == "Upload from PC":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        try:
            temp_pdf_path = f"temp_{uploaded_file.name}"
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.read())
            
            sections = process_pdf(temp_pdf_path)
            if sections:
                store_vectors(sections, uploaded_file.name)
                st.success("PDF uploaded and processed successfully!")
            else:
                st.warning("No content detected in the PDF.")
        except Exception as e:
            st.error(f"Error processing the uploaded PDF: {e}")
else:
    # Document Storage Placeholder (Could be replaced with actual cloud storage functionality)
    st.info("Document Storage feature is currently unavailable.")

# Language Selection
input_language = st.selectbox("Choose Input Language", ("English", "Arabic"))
response_language = st.selectbox("Choose Response Language", ("English", "Arabic"))

# Query Input and Processing
query = st.text_input("Ask a legal question:")

if st.button("Get Answer"):
    if query and uploaded_file:
        response = query_vectors(query, uploaded_file.name)
        st.write(f"**Answer:** {response}")
    else:
        st.warning("Please upload a PDF and enter a query.")
