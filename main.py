import streamlit as st
import pinecone
from deep_translator import GoogleTranslator
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import HuggingFacePipeline

# Initialize Pinecone
pinecone.init(api_key="YOUR_PINECONE_API_KEY", environment="us-west1-gcp")

# Set up embedding model (384-dimensional)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load Pinecone index
index_name = "legal-documents-index"
index = Pinecone.from_existing_index(index_name, embedding_model)

# Function to translate text
def translate_text(text, target_lang):
    return GoogleTranslator(source="auto", target=target_lang).translate(text)

# Function to query Pinecone
def query_vectors(query, pdf_name):
    query_embedding = embedding_model.embed_query(query)
    results = index.similarity_search_by_vector(query_embedding, top_k=3)
    
    # Extract clean legal response
    if results:
        response_texts = [res.page_content for res in results if pdf_name in res.metadata.get("source", "")]
        return response_texts[0] if response_texts else "No relevant article found."
    return "No relevant article found."

# Streamlit UI
st.title("📜 AI-Powered Legal HelpDesk for Saudi Arabia")

# Sidebar for stored PDFs
st.sidebar.header("📂 Stored PDFs")
stored_pdfs = ["Basic Law Governance.pdf", "Law of the Consultative Council.pdf", "Law of the Council of Ministers.pdf"]
for pdf in stored_pdfs:
    st.sidebar.markdown(f"📄 {pdf}")

# Select PDF source
pdf_source = st.radio("Select PDF Source", ["Upload from PC", "Choose from the Document Storage"])

# Select PDF
selected_pdf = None
if pdf_source == "Choose from the Document Storage":
    selected_pdf = st.selectbox("Select a PDF", stored_pdfs)

# Language selection
input_lang = st.radio("Choose Input Language", ["English", "Arabic"], index=0)
response_lang = st.radio("Choose Response Language", ["English", "Arabic"], index=0)

# Query input
query = st.text_input("Ask a question (in English or Arabic):")

# Process Query
if st.button("Get Answer"):
    if selected_pdf and query:
        detected_lang = GoogleTranslator(source="auto", target="en").translate(query)
        response = query_vectors(detected_lang, selected_pdf)

        # Display response in selected language
        if response_lang == "Arabic":
            response = translate_text(response, "ar")
            st.markdown(f"<div dir='rtl' style='text-align: right;'>{response}</div>", unsafe_allow_html=True)
        else:
            st.write(f"**Answer:** {response}")
    else:
        st.warning("Please enter a query and select a PDF.")
