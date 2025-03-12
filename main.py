import os
import re
import pinecone
import pdfplumber
import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# Set up environment variables and initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)

# Initialize Pinecone Index
index_name = "helpdesk"
index = pc.Index(index_name)

# Load the Sentence-Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract content from PDF
def extract_pdf_content(pdf_path):
    chapters = []
    articles = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                # Extract chapters and articles using simple heuristics
                chapter_match = re.search(r'Chapter (\d+)', text)
                article_match = re.search(r'Article (\d+|[A-Za-z]+)', text)

                if chapter_match:
                    chapters.append({"content": text, "chapter": chapter_match.group(1)})

                if article_match:
                    articles.append({"content": text, "title": article_match.group(0), "chapter": chapter_match.group(1) if chapter_match else None})

    return chapters, articles

# Function to store vectors in Pinecone
def store_vectors(chapters, articles, pdf_name):
    try:
        # Store chapters
        for i, chapter in enumerate(chapters):
            vector = model.encode(chapter['content']).tolist()
            print(f"Storing chapter {i}: {vector[:5]}...")  # Debug print
            index.upsert([(
                f"{pdf_name}-chapter-{i}", vector, 
                {"pdf_name": pdf_name, "text": chapter['content'], "type": "chapter"}
            )])

        # Store articles
        for i, article in enumerate(articles):
            # Extract article number correctly
            article_number_match = re.search(r'Article (\d+|[A-Za-z]+)', article['title'], re.IGNORECASE)
            if article_number_match:
                article_number = article_number_match.group(1)
            else:
                article_number = str(i)  # Fallback to index if no article number is found

            vector = model.encode(article['content']).tolist()
            print(f"Storing article {article_number}: {vector[:5]}...")  # Debug print
            index.upsert([(
                f"{pdf_name}-article-{article_number}", vector, 
                {"pdf_name": pdf_name, "chapter": article['chapter'], "text": article['content'], "type": "article", "title": article['title']}
            )])

        print(f"Data successfully stored for {pdf_name}")
    except Exception as e:
        print(f"Error while storing to Pinecone: {e}")

# Streamlit UI to upload PDF files
st.title("Legal Document Storage and Search Bot")

uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True, type="pdf")

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.write(f"Processing file: {uploaded_file.name}")

        # Save the uploaded file temporarily
        with open(f'/tmp/{uploaded_file.name}', 'wb') as f:
            f.write(uploaded_file.read())

        # Extract content from the PDF
        pdf_path = f'/tmp/{uploaded_file.name}'
        chapters, articles = extract_pdf_content(pdf_path)

        # Store the extracted chapters and articles in Pinecone
        store_vectors(chapters, articles, uploaded_file.name)

        # Optionally, display content in Streamlit
        st.write(f"Extracted Chapters: {len(chapters)}")
        st.write(f"Extracted Articles: {len(articles)}")

        # Provide feedback to user
        st.success(f"File {uploaded_file.name} has been processed and stored successfully!")

# Optionally, search the stored data
query = st.text_input("Enter your legal query:")
if query:
    # Encode the query and perform a search in Pinecone
    query_vector = model.encode(query).tolist()

    # Query Pinecone for similar content
    results = index.query(query_vector, top_k=5)

    if results:
        st.write("Top 5 Results:")
        for result in results['matches']:
            st.write(f"ID: {result['id']}, Score: {result['score']}")
            st.write(f"Text: {result['metadata']['text'][:500]}...")  # Display first 500 characters
    else:
        st.write("No results found.")
