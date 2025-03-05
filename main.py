import streamlit as st
import pinecone
import PyPDF2
import os
import pdfplumber
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import re

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
index_name = "helpdesk"

from pinecone import Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(index_name)

model = SentenceTransformer("all-MiniLM-L6-v2")

# Process PDF and extract text article-wise
def process_pdf(pdf_path, chunk_size=500):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    
    # Clean up text by removing unnecessary line breaks and excessive spaces
    text = re.sub(r'\n+', ' ', text)  # Remove excessive line breaks
    text = re.sub(r'\s{2,}', ' ', text)  # Remove excessive spaces
    
    print(f"Extracted Text: {text[:1000]}")  # Print first 1000 characters to check
    
    if not text:
        raise ValueError("No text extracted from PDF. Please check the PDF file.")
    
    # Split the text into articles based on "Article X" pattern (adjust pattern if needed)
    articles = re.split(r'(Article \d+:)', text)
    
    # Create chunks for each article
    article_chunks = []
    current_article = None
    
    for part in articles:
        if part.strip().startswith("Article"):
            if current_article:
                article_chunks.append(current_article)  # Save the previous article
            current_article = {'title': part.strip(), 'content': ""}
        elif current_article:
            current_article['content'] += part.strip()  # Append content to current article
    
    # Don't forget to add the last article
    if current_article:
        article_chunks.append(current_article)
    
    return article_chunks

# Store the extracted articles in Pinecone
def store_vectors(article_chunks, pdf_name):
    for i, article in enumerate(article_chunks):
        # Store title and content as metadata
        vector = model.encode(article['content']).tolist()
        index.upsert([(f"{pdf_name}-article-{i}", vector, {"pdf_name": pdf_name, "title": article['title'], "content": article['content']})])

# Query the stored vectors
def query_vectors(query, selected_pdf):
    vector = model.encode(query).tolist()
    results = index.query(vector=vector, top_k=5, include_metadata=True, filter={"pdf_name": {"$eq": selected_pdf}})
    
    if results["matches"]:
        matched_texts = [match["metadata"]["content"] for match in results["matches"]]
        return "\n\n".join(matched_texts)
    else:
        return "No relevant information found in the selected document."

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>AI-Powered Legal HelpDesk</h1>", unsafe_allow_html=True)

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    temp_pdf_path = f"temp_{uploaded_file.name}"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    article_chunks = process_pdf(temp_pdf_path)
    store_vectors(article_chunks, uploaded_file.name)
    st.success("PDF uploaded and processed!")

# Query input
query = st.text_input("Ask a legal question:")
if st.button("Get Answer"):
    if uploaded_file and query:
        response = query_vectors(query, uploaded_file.name)
        st.write(f"**Answer:** {response}")
    else:
        st.warning("Please upload a PDF and enter a query.")
