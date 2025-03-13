st.title("📖 AI-Powered Legal HelpDesk")

# Upload PDFs
uploaded_files = st.file_uploader("📂 Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    pdf_texts = extract_text_from_pdfs(uploaded_files)
    store_vectors(pdf_texts)

# Query Pinecone
query = st.text_input("🔎 Enter your legal question")
if query:
    query_pinecone(query)
