import chromadb
from transformers import pipeline

client = chromadb.Client()

try:
    collection = client.get_collection('resume_embeddings')
except chromadb.errors.NotFoundError:
    collection = client.create_collection(name='resume_embeddings')

generator = pipeline("text-generation", model="gpt2", device=-1)

def retriever(query, top_k=3):
    """
    Retrieves top-k similar resumes from the ChromaDB collection.
    """
    results = collection.query(query_texts=[query], n_results=top_k)
    return results['documents'], results['metadatas']

def answer(query):
    """
    Generates constructive resume feedback using retrieved resumes for context.
    """
    retrieved_docs, metadatas = retriever(query)
    context = "\n\n---\n\n".join([item for sublist in retrieved_docs for item in sublist])

    # Detailed prompt for feedback
    prompt = f"""
You are an expert resume coach. You will analyze a candidate's resume and provide constructive feedback to help improve it.

Here is the resume from a candidate:
{query}

Below are similar resumes from other candidates who applied for similar roles:
{context}

Based on this, provide detailed feedback on the following aspects of the resume:
1. **Formatting**: Is the resume visually appealing and easy to follow? Are there any areas where the layout can be improved for better readability?
2. **Wording**: Are there any areas where the language can be more impactful? Is the resume concise, clear, and professional?
3. **Missing Details**: Are there any important skills, experiences, or qualifications that are missing from the resume?
4. **Alignment with Job Goals**: Does the resume align well with the candidate’s likely career path or job role?

Be specific, helpful, and concise in your feedback. Focus on the overall impression of the resume and offer concrete suggestions for improvement.
"""

    response = generator(prompt, max_length=1024, truncation=True, do_sample=True, temperature=0.7, max_new_tokens=150)

    return response[0]['generated_text']
