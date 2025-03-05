import streamlit as st
import pinecone
import PyPDF2
import os
import re
import time
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer

def initialize_pinecone():
    try:
        # Securely retrieve Pinecone API key
        PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY")
        if not PINECONE_API_KEY:
            st.error("Pinecone API Key is missing. Please configure it in Streamlit secrets.")
            return None

        # Initialize Pinecone with error handling
        pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
        
        # Use a more robust index name
        index_name = "legal-helpdesk-index"

        # Check and create index with error handling
        try:
            if index_name not in pc.list_indexes().names():
                st.info(f"Creating Pinecone index: {index_name}")
                pc.create_index(
                    name=index_name,
                    dimension=384,  # Adjusted to match the SentenceTransformer model
                    metric="cosine"
                )
                time.sleep(5)  # Wait for index initialization
        except Exception as e:
            st.error(f"Error creating Pinecone index: {e}")
            return None

        # Initialize index
        try:
            index = pc.Index(index_name)
            st.success("Pinecone Index initialized successfully")
            return index
        except Exception as e:
            st.error(f"Error initializing Pinecone index: {e}")
            return None

    except Exception as e:
        st.error(f"Pinecone initialization error: {e}")
        return None

def process_pdf(uploaded_file):
    try:
        # Use BytesIO for in-memory file handling
        from io import BytesIO
        
        reader = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

        # More robust chapter splitting
        chapters = re.split(r'(CHAPTER\s+(?:ONE|[0-9]+):|ARTICLE\s+[0-9]+:)', text, flags=re.IGNORECASE)
        structured_data = {}

        for i in range(1, len(chapters), 2):
            chapter_title = chapters[i].strip()
            chapter_content = chapters[i + 1].strip() if i + 1 < len(chapters) else ""
            
            if chapter_content:
                structured_data[chapter_title] = chapter_content

        return structured_data
    except Exception as e:
        st.error(f"PDF processing error: {e}")
        return {}

def get_existing_pdfs(index):
    try:
        # More robust method to retrieve existing PDFs
        results = index.query(
            vector=[0]*384,  # Zero vector matching model dimension
            top_k=1000,
            include_metadata=True
        )
        return {match['metadata'].get('pdf_name', '') for match in results.get('matches', [])}
    except Exception as e:
        st.error(f"Error retrieving existing PDFs: {e}")
        return set()

def store_vectors(index, structured_data, pdf_name, embedder):
    try:
        # Check if PDF already exists
        existing_pdfs = get_existing_pdfs(index)
        if pdf_name in existing_pdfs:
            st.warning(f"{pdf_name} already exists in Pinecone. Skipping storage.")
            return

        # Store each chapter as a vector
        for title, content in structured_data.items():
            try:
                vector = embedder.encode(content).tolist()
                metadata = {
                    "pdf_name": pdf_name,
                    "chapter": title,
                    "text": content
                }
                vector_id = f"{pdf_name}-{title}"
                index.upsert([(vector_id, vector, metadata)])
            except Exception as chapter_error:
                st.error(f"Error storing chapter {title}: {chapter_error}")

        st.success(f"Successfully stored vectors for {pdf_name}")
    except Exception as e:
        st.error(f"Vector storage error: {e}")

def query_vectors(index, embedder, query, selected_pdf):
    try:
        # Enhanced query processing
        vector = embedder.encode(query).tolist()
        
        results = index.query(
            vector=vector, 
            top_k=5, 
            include_metadata=True, 
            filter={"pdf_name": {"$eq": selected_pdf}}
        )

        if not results.get('matches'):
            return "No relevant information found in the document."

        # Return the most relevant match
        best_match = results['matches'][0]
        return f"**Extracted Answer:**\n\n{best_match['metadata'].get('text', '')}"

    except Exception as e:
        st.error(f"Query processing error: {e}")
        return f"Error processing query: {e}"

def main():
    st.markdown("<h1 style='text-align: center;'>📜 AI-Powered Legal HelpDesk</h1>", unsafe_allow_html=True)

    # Initialize embedder outside of function to avoid repeated loading
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    # Initialize Pinecone
    index = initialize_pinecone()
    if not index:
        st.stop()

    # PDF Upload and Processing
    uploaded_file = st.file_uploader("📂 Upload a PDF", type=["pdf"])
    if uploaded_file:
        structured_data = process_pdf(uploaded_file)
        store_vectors(index, structured_data, uploaded_file.name, embedder)

    # Language Selection and Query
    st.write("### Query Document")
    selected_pdf = st.selectbox("Select PDF", list(get_existing_pdfs(index)) or ["No PDFs uploaded"])
    
    if selected_pdf == "No PDFs uploaded":
        st.warning("Please upload a PDF first.")
        return

    query = st.text_input("Ask a question about the document")
    
    if st.button("Search Document"):
        if query and selected_pdf:
            response = query_vectors(index, embedder, query, selected_pdf)
            st.markdown(response)

if __name__ == "__main__":
    main()
