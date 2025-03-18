import os
import streamlit as st
import pinecone
import PyPDF2

# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "helpdesk"

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Streamlit UI
st.title("📜 AI-Powered Legal HelpDesk for Saudi Arabia")

# Sidebar for Uploaded PDFs
st.sidebar.title("📂 Uploaded PDFs")

# Ensure session state stores PDFs
if "uploaded_pdfs" not in st.session_state:
    st.session_state["uploaded_pdfs"] = {}

# Load PDFs stored in Pinecone
pinecone_docs = set()
query_results = index.query(queries=[[0] * 384], top_k=50, include_metadata=True)  # Dummy query to fetch stored PDFs
if query_results and query_results.get('results'):
    for match in query_results['results'][0]['matches']:
        pdf_name = match['metadata'].get("pdf_name")
        if pdf_name:
            pinecone_docs.add(pdf_name)

# Combine session PDFs and Pinecone PDFs
all_pdfs = set(st.session_state["uploaded_pdfs"].keys()).union(pinecone_docs)

if all_pdfs:
    selected_pdf = st.sidebar.selectbox("Select a PDF:", list(all_pdfs))
    st.session_state["selected_pdf"] = selected_pdf
else:
    st.sidebar.warning("No PDFs uploaded yet.")

# File Upload Section
st.subheader("📂 Upload a PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file:
    pdf_name = uploaded_file.name
    st.session_state["uploaded_pdfs"][pdf_name] = uploaded_file

    # Read and store the PDF content
    with open(f"temp_{pdf_name}", "wb") as f:
        f.write(uploaded_file.getbuffer())

    with open(f"temp_{pdf_name}", "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

    # Store in Pinecone
    index.upsert([(pdf_name, [0] * 384, {"pdf_name": pdf_name, "content": text})])

    st.success(f"✅ {pdf_name} uploaded and stored successfully!")

# Language Selection
st.subheader("🌍 Choose Input & Response Language")
input_lang = st.radio("Choose Input Language:", ["English", "Arabic"])
response_lang = st.radio("Choose Response Language:", ["English", "Arabic"])

# Question Input
st.subheader("💬 Ask a Legal Question")
question = st.text_input("Type your question here...")

if st.button("Submit"):
    if not question:
        st.warning("⚠️ Please enter a question.")
    elif "selected_pdf" not in st.session_state:
        st.warning("⚠️ Please select or upload a PDF.")
    else:
        # Perform Pinecone search
        search_results = index.query(queries=[[0] * 384], top_k=5, include_metadata=True)
        response_text = "\n".join(
            [match["metadata"].get("content", "")[:500] for match in search_results["results"][0]["matches"]]
        )

        # Display AI response
        st.subheader("📜 AI-Generated Response")
        st.write(response_text if response_text else "No relevant legal information found.")

