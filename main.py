import streamlit as st
import os
import PyPDF2
import pickle
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator
from sklearn.metrics.pairwise import cosine_similarity


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


PDF_STORAGE_DIR = "pdf_repository"
if not os.path.exists(PDF_STORAGE_DIR):
    os.makedirs(PDF_STORAGE_DIR)


def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    return text


def store_pdf_embeddings(pdf_name, pdf_text):
    embeddings = model.encode(pdf_text, convert_to_tensor=False)
    storage_path = os.path.join(PDF_STORAGE_DIR, f"{pdf_name}.pkl")
    with open(storage_path, "wb") as f:
        pickle.dump({"text": pdf_text, "embeddings": embeddings}, f)


def is_pdf_stored(pdf_name):
    return os.path.exists(os.path.join(PDF_STORAGE_DIR, f"{pdf_name}.pkl"))


def load_pdf_embeddings(pdf_name):
    storage_path = os.path.join(PDF_STORAGE_DIR, f"{pdf_name}.pkl")
    with open(storage_path, "rb") as f:
        return pickle.load(f)


st.markdown("<h1 style='text-align: center;'>AI-Powered Legal HelpDesk</h1>", unsafe_allow_html=True)


st.sidebar.header("📂 Stored PDFs")
stored_pdfs = [f.replace(".pkl", "") for f in os.listdir(PDF_STORAGE_DIR) if f.endswith(".pkl")]
if stored_pdfs:
    selected_pdf = st.sidebar.selectbox("Choose a stored PDF", stored_pdfs)
else:
    selected_pdf = None


st.header("📂 Upload a PDF")
uploaded_file = st.file_uploader("Upload a legal document (PDF)", type=["pdf"])

if uploaded_file:
    pdf_name = uploaded_file.name
    pdf_path = os.path.join(PDF_STORAGE_DIR, pdf_name)

    if not is_pdf_stored(pdf_name):
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        pdf_text = extract_text_from_pdf(pdf_path)
        store_pdf_embeddings(pdf_name, pdf_text)
        st.success("PDF uploaded, processed, and stored!")
    else:
        st.info("PDF is already stored.")


input_lang = st.radio("Choose Input Language", ["English", "Arabic"], index=0)
response_lang = st.radio("Choose Response Language", ["English", "Arabic"], index=0)


if input_lang == "Arabic":
    query = st.text_input("اسأل سؤالاً (باللغة العربية أو الإنجليزية):")
    st.markdown("<style>.stTextInput>div>div>input {direction: rtl; text-align: right;}</style>", unsafe_allow_html=True)
else:
    query = st.text_input("Ask a question (in English or Arabic):")


if st.button("Get Answer"):
    if selected_pdf and query:
        # Load stored PDF text and embeddings
        pdf_data = load_pdf_embeddings(selected_pdf)
        pdf_text = pdf_data["text"]
        pdf_embeddings = pdf_data["embeddings"]

     
        query_embedding = model.encode(query, convert_to_tensor=False)

       
        similarity_scores = cosine_similarity([query_embedding], [pdf_embeddings])[0]
        best_match_index = similarity_scores.argmax()

       
        matched_text = pdf_text.split("\n")[best_match_index]

    
        if response_lang == "Arabic":
            matched_text = GoogleTranslator(source="auto", target="ar").translate(matched_text)
            st.markdown(f"<div dir='rtl' style='text-align: right;'>{matched_text}</div>", unsafe_allow_html=True)
        else:
            st.write(f"**Answer:** {matched_text}")
    else:
        st.warning("Please enter a query and select a stored PDF.")
