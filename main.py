import streamlit as st
import pinecone
import PyPDF2
import re
import numpy as np
import traceback
from sentence_transformers import SentenceTransformer

class PDFChunker:
    def __init__(self, embedder_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(embedder_model)
    
    def preprocess_text(self, text):
        """
        Clean and normalize text
        """
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers
        text = re.sub(r'\b\d+\b', '', text)
        
        # Remove multiple newlines
        text = re.sub(r'\n+', '\n', text)
        
        return text.strip()
    
    def chunk_text(self, text, max_chunk_size=500, overlap=100):
        """
        Split text into semantic chunks with overlap
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_length += len(sentence)
            
            # If chunk is too large, create a new chunk
            if current_length > max_chunk_size:
                # Join chunk and add to chunks
                full_chunk = ' '.join(current_chunk)
                chunks.append(full_chunk)
                
                # Prepare next chunk with overlap
                current_chunk = current_chunk[-int(overlap/50):]
                current_length = len(' '.join(current_chunk))
        
        # Add last chunk if not empty
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def process_pdf(self, uploaded_file):
        """
        Extract and chunk PDF content
        """
        try:
            from io import BytesIO
            
            # Read PDF
            reader = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
            
            # Extract full text
            full_text = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text.append(self.preprocess_text(page_text))
            
            # Combine and chunk text
            combined_text = ' '.join(full_text)
            chunks = self.chunk_text(combined_text)
            
            # Organize chunks with metadata
            structured_chunks = []
            for i, chunk in enumerate(chunks):
                structured_chunks.append({
                    'id': f'chunk_{i}',
                    'text': chunk,
                    'embedding': self.embedder.encode(chunk).tolist()
                })
            
            return structured_chunks
        
        except Exception as e:
            st.error(f"PDF Processing Error: {e}")
            traceback.print_exc()
            return []

class PineconeVectorStore:
    def __init__(self, api_key, index_name="legal-document-index"):
        self.pc = pinecone.Pinecone(api_key=api_key)
        self.index_name = index_name
        self.dimension = 384  # Match embedding dimension
    
    def create_index(self):
        """
        Create Pinecone index if not exists
        """
        try:
            if self.index_name not in self.pc.list_indexes().names():
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine"
                )
            return self.pc.Index(self.index_name)
        except Exception as e:
            st.error(f"Pinecone Index Creation Error: {e}")
            return None
    
    def store_document(self, index, pdf_name, chunks):
        """
        Store document chunks in Pinecone
        """
        try:
            for chunk in chunks:
                metadata = {
                    'pdf_name': pdf_name,
                    'text': chunk['text']
                }
                index.upsert([
                    (chunk['id'], chunk['embedding'], metadata)
                ])
            st.success(f"Stored {len(chunks)} chunks for {pdf_name}")
        except Exception as e:
            st.error(f"Vector Storage Error: {e}")
    
    def semantic_search(self, index, embedder, query, pdf_name, top_k=3):
        """
        Perform semantic search across document chunks
        """
        try:
            # Embed query
            query_embedding = embedder.encode(query).tolist()
            
            # Perform similarity search
            results = index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter={"pdf_name": {"$eq": pdf_name}}
            )
            
            # Format response
            response = "### Relevant Passages:\n\n"
            for match in results['matches']:
                metadata = match.get('metadata', {})
                response += f"**Relevance Score: {match['score']:.2%}**\n"
                response += f"{metadata.get('text', 'No text found')}\n\n---\n\n"
            
            return response
        except Exception as e:
            st.error(f"Semantic Search Error: {e}")
            return f"Search Error: {e}"

def main():
    st.title("📄 Advanced PDF Search Engine")
    
    # Initialize components
    chunker = PDFChunker()
    
    # Pinecone API Key (replace with your method of key retrieval)
    PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY")
    if not PINECONE_API_KEY:
        st.error("Pinecone API Key is required!")
        return
    
    vector_store = PineconeVectorStore(PINECONE_API_KEY)
    index = vector_store.create_index()
    
    if not index:
        st.error("Failed to create Pinecone index")
        return
    
    # PDF Upload
    uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])
    
    if uploaded_file:
        # Process PDF
        chunks = chunker.process_pdf(uploaded_file)
        
        # Store in Pinecone
        vector_store.store_document(index, uploaded_file.name, chunks)
        
        # Preview chunks
        st.write("### Extracted Text Chunks:")
        for chunk in chunks[:5]:
            st.text_area(f"Chunk {chunk['id']}", chunk['text'], height=100)
    
    # Query Interface
    st.header("Search Document")
    query = st.text_input("Ask a question about the document")
    
    if st.button("Search"):
        if query and uploaded_file:
            # Perform semantic search
            results = vector_store.semantic_search(
                index, 
                chunker.embedder, 
                query, 
                uploaded_file.name
            )
            st.markdown(results)

if __name__ == "__main__":
    main()
