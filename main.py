from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "helpdesk"  # Your Pinecone index name

# Initialize the SentenceTransformer model to convert queries into vectors
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Access the Pinecone index
index = pc.Index(index_name)

def index_data_to_pinecone(data):
    """Function to index the data in Pinecone."""
    for item in data:
        # You need to make sure the vector for each item is generated and added
        vector = model.encode(item['text']).tolist()
        
        # Index the data with metadata, and here we assume 'id' is unique for each document
        index.upsert(
            vectors=[
                {
                    "id": item['id'],
                    "values": vector,
                    "metadata": {
                        "pdf_name": item['pdf_name'],
                        "title": item['title'],
                        "text": item['text'],
                        "chapter": item['chapter'],
                        "type": item['type']
                    }
                }
            ]
        )

def query_vectors(query, selected_pdf):
    """Queries Pinecone for the most relevant result, prioritizing exact article matches."""
    # Encode the query into a vector
    query_vector = model.encode(query).tolist()
    
    # Search for matching articles based on metadata filters
    results = index.query(
        vector=query_vector,  # Provide the query vector to match against
        top_k=5,  # Get top 5 results
        include_metadata=True,
        filter={"pdf_name": {"$eq": selected_pdf}},  # Ensure pdf_name is the same as the selected one
        # You can add more filters for article, chapter, etc., if necessary
    )
    
    if results and results["matches"]:
        return "\n\n".join([match["metadata"]["text"] for match in results["matches"]])
    return "No relevant answer found."

# Example data to index
data_to_index = [
    {
        "id": "1",
        "pdf_name": "Basic Law Governance.pdf",
        "title": "Article 1",
        "text": "The Kingdom of Saudi Arabia is a sovereign Arab Islamic State. Its religion is Islam. Its constitution is Almighty God's Book, The Holy Qur'an, and the Sunna (Traditions) of the Prophet (PBUH). Arabic is the language of the Kingdom. The City of Riyadh is the capital.",
        "chapter": "Chapter One: General Principles",
        "type": "article"
    },
    {
        "id": "2",
        "pdf_name": "Basic Law Governance.pdf",
        "title": "Article 2",
        "text": "The State public holidays are Eid Al Fitr (the Feast of Ramadan) and Eid Al Adha (The Feast of the Sacrifice). Its calendar follows the Hijri year (the lunar year).",
        "chapter": "Chapter One: General Principles",
        "type": "article"
    }
]

# Index the example data to Pinecone
index_data_to_pinecone(data_to_index)

# Example query
query = "What are the public holidays in Saudi Arabia?"
selected_pdf = "Basic Law Governance.pdf"

# Query Pinecone with the encoded vector and metadata filter
response = query_vectors(query, selected_pdf)

# Print the response from Pinecone
print(response)
