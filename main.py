import streamlit as st
import pinecone
import PyPDF2
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import re

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
index_name = "helpdesk"

from pinecone import Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(index_name)

model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to process the PDF and extract title and content
def process_pdf(pdf_path, chunk_size=500):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
    # Assume titles are lines with a particular format (like 'Title: <something>')
    # You can refine this with a more complex regex if necessary
    sections = []
    current_title = None
    current_content = []
    
    # Split text into paragraphs or sections (here we assume titles appear before paragraphs)
    paragraphs = text.split('\n')
    
    for para in paragraphs:
        # Check if the paragraph looks like a title (you can adjust this regex to suit your needs)
        if re.match(r'^[A-Z][A-Za-z0-9\s]+$', para.strip()):  # Adjust regex for title pattern
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

# Store vectors of title-content pairs
def store_vectors(sections, pdf_name):
    for i, section in enumerate(sections):
        title = section['title']
        content = section['content']
        
        # Create a vector for both title and content
        title_vector = model.encode(title).tolist()
        content_vector = model.encode(content).tolist()
        
        # Store title and content vectors in Pinecone
        index.upsert([
            (f"{pdf_name}-title-{i}", title_vector, {"pdf_name": pdf_name, "text": title, "type": "title"}),
            (f"{pdf_name}-content-{i}", content_vector, {"pdf_name": pdf_name, "text": content, "type": "content"})
        ])

# Query for relevant articles
def query_vectors(query, selected_pdf):
    vector = model.encode(query).tolist()
    results = index.query(vector=vector, top_k=5, include_metadata=True, filter={"pdf_name": {"$eq": selected_pdf}})
    
    if results["matches"]:
        matched_texts = [match["metadata"]["text"] for match in results["matches"]]
        return "\n\n".join(matched_texts)
    else:
        return "No relevant information found in the selected document."

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>AI-Powered Legal HelpDesk</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    temp_pdf_path = f"temp_{uploaded_file.name}"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    sections = process_pdf(temp_pdf_path)
    store_vectors(sections, uploaded_file.name)
    st.success("PDF uploaded and processed!")

query = st.text_input("Ask a legal question:")
if st.button("Get Answer"):
    if uploaded_file and query:
        response = query_vectors(query, uploaded_file.name)
        st.write(f"**Answer:** {response}")
    else:
        st.warning("Please upload a PDF and enter a query.")
