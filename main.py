import streamlit as st
import PyPDF2
import numpy as np
import os
import re
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator

class SaudiLegalHelpDesk:
    def __init__(self):
        # Initialize multilingual embedding model
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # Create document storage directory if not exists
        self.storage_dir = "document_storage"
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Initialize document storage
        self.document_storage = self._get_stored_documents()
    
    def _get_stored_documents(self):
        """Retrieve list of documents in storage directory"""
        try:
            return [f for f in os.listdir(self.storage_dir) if f.endswith('.pdf')]
        except Exception as e:
            st.error(f"Error accessing document storage: {e}")
            return []
    
    def save_uploaded_document(self, uploaded_file):
        """Save uploaded document to storage"""
        try:
            file_path = os.path.join(self.storage_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Refresh document storage
            self.document_storage = self._get_stored_documents()
            
            st.success(f"Document {uploaded_file.name} saved successfully!")
            return file_path
        except Exception as e:
            st.error(f"Error saving document: {e}")
            return None
    
    def extract_text(self, file_path):
        """Extract text from PDF with robust error handling"""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ' '.join([
                    page.extract_text() for page in reader.pages 
                    if page.extract_text()
                ])
            return self.preprocess_text(text)
        except Exception as e:
            st.error(f"Error extracting PDF text: {e}")
            return ""
    
    def preprocess_text(self, text):
        """Clean and prepare text for analysis"""
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers
        text = re.sub(r'\b\d+\b', '', text)
        
        # Remove headers/footers (optional)
        text = re.sub(r'^.*\n', '', text)
        
        return text.strip()
    
    def split_into_sections(self, text):
        """Split text into meaningful sections"""
        # Split by Articles or Chapters
        sections = re.split(r'(Article\s+\d+:|Chapter\s+\d+:)', text)
        
        # Clean and filter sections
        cleaned_sections = []
        for i in range(1, len(sections), 2):
            if i+1 < len(sections):
                section = {
                    'title': sections[i].strip(),
                    'content': sections[i+1].strip()
                }
                cleaned_sections.append(section)
        
        return cleaned_sections
    
    def find_most_relevant_section(self, query, sections):
        """Advanced semantic search with contextual understanding"""
        try:
            # Encode query
            query_embedding = self.model.encode(query)
            
            # Score sections
            section_scores = []
            for section in sections:
                # Combine title and content for embedding
                full_text = f"{section['title']} {section['content']}"
                section_embedding = self.model.encode(full_text)
                
                # Calculate similarity
                similarity = np.dot(query_embedding, section_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(section_embedding)
                )
                
                section_scores.append({
                    'section': section,
                    'score': similarity
                })
            
            # Sort and get top matching section
            top_section = max(section_scores, key=lambda x: x['score'])
            
            # Generate detailed response
            response = self._generate_comprehensive_response(
                query, 
                top_section['section'], 
                top_section['score']
            )
            
            return response
        
        except Exception as e:
            st.error(f"Relevance search error: {e}")
            return "Unable to process the query."
    
    def _generate_comprehensive_response(self, query, section, relevance_score):
        """Generate a scholarly, context-rich response"""
        response = f"**Contextual Legal Analysis**\n\n"
        response += f"**Section:** {section['title']}\n\n"
        response += f"**Relevance Score:** {relevance_score:.2%}\n\n"
        response += f"**Key Insights:**\n{section['content'][:1000]}...\n\n"
        response += "**Legal Interpretation:**\n"
        response += "This section provides critical insights into the legal framework, "
        response += "highlighting the nuanced interpretations and regulatory mechanisms "
        response += "within the Saudi Arabian legal system.\n"
        
        return response
    
    def translate_response(self, response, target_language):
        """Translate response if needed"""
        if target_language.lower() == 'arabic':
            try:
                translator = GoogleTranslator(source='auto', target='ar')
                return translator.translate(response)
            except Exception as e:
                st.error(f"Translation error: {e}")
                return response
        return response

def main():
    # Streamlit page configuration
    st.set_page_config(
        page_title="Saudi Legal HelpDesk", 
        page_icon="⚖️", 
        layout="wide"
    )
    
    # Custom CSS for enhanced styling
    st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        color: #2C3E50;
        text-align: center;
        margin-bottom: 20px;
    }
    .section-header {
        color: #34495E;
        border-bottom: 2px solid #3498DB;
        padding-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.markdown("<h1 class='main-title'>🏛️ AI-Powered Legal HelpDesk for Saudi Arabia</h1>", unsafe_allow_html=True)
    
    # Initialize helpdesk
    helpdesk = SaudiLegalHelpDesk()
    
    # Main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<h2 class='section-header'>📄 Document Source</h2>", unsafe_allow_html=True)
        
        # PDF Source Selection
        source_type = st.radio(
            "Choose Document Source", 
            ["Upload New Document", "Use Existing Document"], 
            horizontal=True
        )
        
        # Document Selection Logic
        if source_type == "Upload New Document":
            uploaded_file = st.file_uploader(
                "Upload PDF", 
                type=['pdf'], 
                help="Upload a legal document in PDF format"
            )
            
            if uploaded_file:
                # Save uploaded document
                file_path = helpdesk.save_uploaded_document(uploaded_file)
        else:
            file_path = os.path.join(
                helpdesk.storage_dir, 
                st.selectbox("Select Document", helpdesk.document_storage)
            )
    
    with col2:
        st.markdown("<h2 class='section-header'>🌐 Language Settings</h2>", unsafe_allow_html=True)
        
        # Language Selection
        input_lang = st.radio("Input Language", ["English", "Arabic"])
        response_lang = st.radio("Response Language", ["English", "Arabic"])
    
    # Query Interface
    st.markdown("<h2 class='section-header'>❓ Ask Your Legal Question</h2>", unsafe_allow_html=True)
    
    query = st.text_input(
        f"Enter your legal query in {input_lang}", 
        placeholder="Type your legal question here..."
    )
    
    # Search Button
    if st.button("Analyze Document", type="primary"):
        if 'file_path' in locals() and file_path and query:
            try:
                # Extract full text
                full_text = helpdesk.extract_text(file_path)
                
                # Split into sections
                document_sections = helpdesk.split_into_sections(full_text)
                
                # Find most relevant section
                response = helpdesk.find_most_relevant_section(query, document_sections)
                
                # Translate if needed
                final_response = helpdesk.translate_response(response, response_lang)
                
                # Display response
                st.markdown("### 📋 Analysis Result")
                st.markdown(final_response)
            
            except Exception as e:
                st.error(f"Analysis failed: {e}")
        else:
            st.warning("Please upload/select a document and enter a query.")

if __name__ == "__main__":
    main()
