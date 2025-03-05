import streamlit as st
import pinecone
import PyPDF2
import numpy as np
import os
import re
import time
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator

# === 📌 Initialize Pinecone === #
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets.get("PINECONE_ENV", "us-east-1")

# Initialize Pinecone connection
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index_name = "helpdesk"

# Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    print("⚠️ Index does not exist. Creating index...")
    pc.create_index(name=index_name, dimension=1536, metric="cosine")

time.sleep(5)  # Wait for Pinecone index to be ready
index = pc.Index(index_name)
print("✅ Pinecone Index Ready:", index.describe_index_stats())

# === 📌 AI Model === #
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# === 📌 Saudi Legal HelpDesk Class === #
class SaudiLegalHelpDesk:
    def __init__(self):
        self.model = embedder
        self.storage_dir = "document_storage"
        os.makedirs(self.storage_dir, exist_ok=True)
        self.document_storage = self._get_stored_documents()

    def _get_stored_documents(self):
        """Retrieve stored documents in storage directory."""
        try:
            return [f for f in os.listdir(self.storage_dir) if f.endswith('.pdf')]
        except Exception as e:
            st.error(f"Error accessing document storage: {e}")
            return []

    def save_uploaded_document(self, uploaded_file):
        """Save uploaded PDF to storage."""
        try:
            file_path = os.path.join(self.storage_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            self.document_storage = self._get_stored_documents()
            st.success(f"✅ Document '{uploaded_file.name}' saved successfully!")
            return file_path
        except Exception as e:
            st.error(f"Error saving document: {e}")
            return None

    def extract_text(self, file_path):
        """Extract text from PDF."""
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
        """Clean and format text."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\b\d+\b', '', text)  # Remove page numbers
        return text.strip()

    def split_into_sections(self, text):
        """Split text into structured sections."""
        sections = re.split(r'(Article\s+\d+:|Chapter\s+\d+:)', text)
        cleaned_sections = []

        for i in range(1, len(sections), 2):
            if i+1 < len(sections):
                section = {'title': sections[i].strip(), 'content': sections[i+1].strip()}
                cleaned_sections.append(section)

        return cleaned_sections

    def find_most_relevant_section(self, query, sections):
        """Find the best-matching section for a query."""
        try:
            query_embedding = self.model.encode(query)
            section_scores = []

            for section in sections:
                full_text = f"{section['title']} {section['content']}"
                section_embedding = self.model.encode(full_text)

                similarity = np.dot(query_embedding, section_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(section_embedding)
                )

                section_scores.append({'section': section, 'score': similarity})

            top_section = max(section_scores, key=lambda x: x['score'])
            return self._generate_comprehensive_response(query, top_section['section'], top_section['score'])

        except Exception as e:
            st.error(f"Relevance search error: {e}")
            return "Unable to process the query."

    def _generate_comprehensive_response(self, query, section, relevance_score):
        """Generate a well-structured legal answer."""
        response = f"**📜 Legal Analysis**\n\n"
        response += f"**📌 Section:** {section['title']}\n\n"
        response += f"**🔍 Relevance Score:** {relevance_score:.2%}\n\n"
        response += f"**📖 Key Insights:**\n{section['content'][:1000]}...\n\n"
        response += "**⚖️ Legal Interpretation:**\n"
        response += "This section highlights the legal framework, regulations, and interpretations within the Saudi Arabian legal system.\n"
        return response

    def translate_response(self, response, target_language):
        """Translate response if needed."""
        if target_language.lower() == 'arabic':
            try:
                return GoogleTranslator(source='auto', target='ar').translate(response)
            except Exception as e:
                st.error(f"Translation error: {e}")
                return response
        return response

# === 📌 Pinecone Storage Fix === #
def get_existing_pdfs():
    """Retrieve stored PDFs from Pinecone."""
    existing_pdfs = set()
    try:
        results = index.query(vector=[0]*1536, top_k=1000, include_metadata=True)
        for match in results["matches"]:
            pdf_name = match["metadata"].get("pdf_name", "")
            if pdf_name:
                existing_pdfs.add(pdf_name)
    except Exception as e:
        print("⚠️ Error checking existing PDFs:", e)
    
    return existing_pdfs

def store_vectors(structured_data, pdf_name):
    """Store extracted sections in Pinecone, avoiding duplicates."""
    existing_pdfs = get_existing_pdfs()

    if pdf_name in existing_pdfs:
        print(f"⚠️ {pdf_name} already exists in Pinecone. Skipping storage.")
        return
    
    for title, content in structured_data.items():
        vector = embedder.encode(content).tolist()
        metadata = {"pdf_name": pdf_name, "chapter": title, "text": content}
        index.upsert([(f"{pdf_name}-{title}", vector, metadata)])

# === 📌 Streamlit UI === #
def main():
    st.set_page_config(page_title="Saudi Legal HelpDesk", page_icon="⚖️", layout="wide")

    st.markdown("<h1 style='text-align: center;'>🏛️ AI-Powered Legal HelpDesk for Saudi Arabia</h1>", unsafe_allow_html=True)

    helpdesk = SaudiLegalHelpDesk()

    col1, col2 = st.columns([2, 1])

    with col1:
        source_type = st.radio("Choose Document Source", ["Upload New Document", "Use Existing Document"], horizontal=True)

        if source_type == "Upload New Document":
            uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])
            file_path = helpdesk.save_uploaded_document(uploaded_file) if uploaded_file else None
        else:
            file_path = os.path.join(helpdesk.storage_dir, st.selectbox("Select Document", helpdesk.document_storage))

    with col2:
        input_lang = st.radio("Input Language", ["English", "Arabic"])
        response_lang = st.radio("Response Language", ["English", "Arabic"])

    query = st.text_input("🔍 Ask a legal question:")

    if st.button("Analyze Document"):
        if file_path and query:
            full_text = helpdesk.extract_text(file_path)
            document_sections = helpdesk.split_into_sections(full_text)
            response = helpdesk.find_most_relevant_section(query, document_sections)
            final_response = helpdesk.translate_response(response, response_lang)
            st.markdown("### 📋 Analysis Result")
            st.markdown(final_response)

if __name__ == "__main__":
    main()
