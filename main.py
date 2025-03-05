import streamlit as st
import PyPDF2
import pinecone
from deep_translator import GoogleTranslator
from transformers import pipeline

# Initialize Pinecone
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENV"]

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index_name = "helpdesk"
index = pc.Index(index_name)

# Hugging Face Embedding Model
embedding_model = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")

# Function to Process PDF into Chunks
def process_pdf(pdf_path, chunk_size=500):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])

    chunks = []
    words = text.split()  # Ensure words are not split mid-way
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk.strip())
    return chunks

# Store Vectors in Pinecone
def store_vectors(chunks, pdf_name):
    for i, chunk in enumerate(chunks):
        vector = embedding_model(chunk)
        vector = vector[0] if isinstance(vector, list) else vector  # Ensure correct format
        vector = vector.tolist()  # Convert to list of floats
        index.upsert([(f"{pdf_name}-doc-{i}", vector, {"pdf_name": pdf_name, "text": chunk})])

# Query Pinecone for Answers
def query_vectors(query, selected_pdf):
    vector = embedding_model(query)
    vector = vector[0] if isinstance(vector, list) else vector  # Ensure correct format
    vector = vector.tolist()  # Convert to list of floats
    results = index.query(vector=vector, top_k=5, include_metadata=True, filter={"pdf_name": {"$eq": selected_pdf}})

    if results.matches:
        matched_texts = [match.metadata["text"] for match in results.matches]
        return "\n\n".join(matched_texts)
    else:
        return "No relevant information found in the selected document."

# Translate Text
def translate_text(text, target_language):
    return GoogleTranslator(source="auto", target=target_language).translate(text)

# Streamlit UI
st.markdown(
    "<h1 style='text-align: center;'>📜 AI-Powered Legal HelpDesk for Saudi Arabia</h1>",
    unsafe_allow_html=True
)

# Select PDF Source
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    temp_pdf_path = f"temp_{uploaded_file.name}"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    chunks = process_pdf(temp_pdf_path)
    store_vectors(chunks, uploaded_file.name)
    st.success("✅ PDF uploaded and processed successfully!")
    selected_pdf = uploaded_file.name

# Language Selection
input_lang = st.radio("Choose Input Language", ["English", "Arabic"], index=0)
response_lang = st.radio("Choose Response Language", ["English", "Arabic"], index=0)

if input_lang == "Arabic":
    query = st.text_input("📝 اسأل سؤالاً (باللغة العربية أو الإنجليزية):", key="query_input")
    st.markdown(
        "<style>.stTextInput>div>div>input { direction: rtl; text-align: right; }</style>",
        unsafe_allow_html=True
    )
else:
    query = st.text_input("📝 Ask a question (in English or Arabic):", key="query_input")

# Get Answer
if st.button("Get Answer"):
    if uploaded_file and query:
        detected_lang = GoogleTranslator(source="auto", target="en").translate(query)
        response = query_vectors(detected_lang, selected_pdf)

        if response_lang == "Arabic":
            response = translate_text(response, "ar")
            st.markdown(f"<div dir='rtl' style='text-align: right;'>{response}</div>", unsafe_allow_html=True)
        else:
            st.write(f"**Answer:** {response}")
    else:
        st.warning("⚠️ Please enter a query and upload a PDF.")
