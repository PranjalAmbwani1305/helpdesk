import os
import pinecone
import streamlit as st

# Load Pinecone API Key from environment variable
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "helpdesk"

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Function to fetch stored PDFs from Pinecone
def get_stored_pdfs():
    try:
        response = index.describe_index_stats()
        vector_count = response["total_vector_count"]
        
        stored_pdfs = []
        if vector_count > 0:
            for i in range(vector_count):
                vector = index.fetch([str(i)])
                if vector and "metadata" in vector["vectors"][str(i)]:
                    metadata = vector["vectors"][str(i)]["metadata"]
                    if "filename" in metadata:
                        stored_pdfs.append(metadata["filename"])

        return stored_pdfs
    except Exception as e:
        st.error(f"Error fetching stored PDFs: {e}")
        return []

# Fetch stored PDFs dynamically
stored_pdfs = get_stored_pdfs()

# Streamlit UI
st.title("⚖️ AI-Powered Legal HelpDesk for Saudi Arabia")

# Sidebar: Display stored PDFs
st.sidebar.title("📂 Stored PDFs")

if stored_pdfs:
    for pdf in stored_pdfs:
        st.sidebar.markdown(f"📄 {pdf}")
else:
    st.sidebar.write("No PDFs stored yet.")

# File uploader for new PDFs
st.subheader("Upload a PDF")
uploaded_file = st.file_uploader("Drag and drop a file here", type="pdf")

if uploaded_file:
    filename = uploaded_file.name
    file_data = uploaded_file.read()

    # Store metadata in Pinecone
    index.upsert(vectors=[
        {"id": filename, "values": [0.0] * 128, "metadata": {"filename": filename}}
    ])
    
    st.success(f"📄 {filename} has been stored in Pinecone.")

# Input fields for user query
st.subheader("Ask a Legal Question")
st.text_input("Enter your question here:")
st.button("Get Answer")
