import streamlit as st
import pinecone
from sentence_transformers import SentenceTransformer
import json

# Initialize Pinecone
pinecone.init(api_key="YOUR_PINECONE_API_KEY", environment="us-east-1")
index = pinecone.Index("helpdesk")

# Load Embedding Model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Streamlit UI
st.title("Legal Helpdesk AI")

# Sidebar for File Selection
uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
selected_pdf = uploaded_file.name if uploaded_file else ""

if uploaded_file:
    st.sidebar.success(f"Uploaded: {selected_pdf}")

# Function to Store Data in Pinecone
def store_article(pdf_name, chapter_name, article_number, title, content):
    title_vector = model.encode(title).tolist()
    content_vector = model.encode(content).tolist()

    index.upsert([
        (f"{pdf_name}-chapter-{chapter_name}-article-{article_number}-title", title_vector, 
            {"pdf_name": pdf_name, "chapter": chapter_name, "article_number": article_number, "text": title, "type": "title"}),

        (f"{pdf_name}-chapter-{chapter_name}-article-{article_number}-content", content_vector, 
            {"pdf_name": pdf_name, "chapter": chapter_name, "article_number": article_number, "text": content, "type": "content"})
    ])

# Query Function
def query_vectors(query, selected_pdf):
    try:
        vector = model.encode(query).tolist()
        results = index.query(vector=vector, top_k=5, include_metadata=True, filter={"pdf_name": {"$eq": selected_pdf}})
        
        if results["matches"]:
            article_dict = {}  # Store title-content pairs
            
            for match in results["matches"]:
                metadata = match["metadata"]
                text = metadata["text"]
                item_type = metadata["type"]
                chapter = metadata.get("chapter", "Unknown Chapter")
                article_number = metadata.get("article_number", "Unknown Article")

                # If it's an article title, store it as a key
                if item_type == "title":
                    article_dict[text] = {"content": "", "chapter": chapter, "article_number": article_number}
                elif item_type == "content":
                    for title in article_dict.keys():
                        if text.startswith(title):  # Check if content belongs to a title
                            article_dict[title]["content"] = text
                            break
            
            # Prepare response
            response_text = ""
            for title, data in article_dict.items():
                response_text += f"**Chapter: {data['chapter']}**\n"
                response_text += f"**Article {data['article_number']}:**\n"
                response_text += f"{title}\n{data['content']}\n\n"
            
            return response_text if response_text else "No relevant information found in the selected document."
        else:
            return "No relevant information found in the selected document."
    
    except Exception as e:
        return f"Error during query: {e}"

# User Query Input
query = st.text_input("Ask a legal question:")
if query and selected_pdf:
    response = query_vectors(query, selected_pdf)
    st.write(response)
