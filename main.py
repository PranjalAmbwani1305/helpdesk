from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY)

# Corrected function for embedding creation
def store_vectors(chunks, pdf_name):
    for i, chunk in enumerate(chunks):
        response = client.embeddings.create(input=[chunk], model="text-embedding-ada-002")
        vector = response.data[0].embedding
        index.upsert([(f"{pdf_name}-doc-{i}", vector, {"pdf_name": pdf_name, "text": chunk})])

# Corrected function for querying
def query_vectors(query, selected_pdf):
    response = client.embeddings.create(input=[query], model="text-embedding-ada-002")
    vector = response.data[0].embedding
    
    results = index.query(vector=vector, top_k=5, include_metadata=True, filter={"pdf_name": {"$eq": selected_pdf}})

    if results["matches"]:
        matched_texts = [match["metadata"]["text"] for match in results["matches"]]
        combined_text = "\n\n".join(matched_texts)

        prompt = (
            f"Based on the following legal document ({selected_pdf}), provide an accurate and well-reasoned answer:\n\n"
            f"{combined_text}\n\n"
            f"User's Question: {query}"
        )

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI assistant specialized in legal analysis."},
                {"role": "user", "content": prompt}
            ]
        ).choices[0].message.content

        return response
    else:
        return "No relevant information found in the selected document."
