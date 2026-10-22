import os
import sys
import json
import time
import pypdf
import numpy as np
import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types
from groq import Groq
import random
 
 
def extract_text(pdf_path):
    """Extract all text from a PDF and return it as a single string."""
    reader = pypdf.PdfReader(pdf_path)
    pages = [page.extract_text() for page in reader.pages]
    return "\n".join(pages)
 
 
def chunk_text(text, chunk_size=500, overlap=100):
    """Split text into overlapping chunks of fixed size."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    
    chunks = []
    
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i : i + chunk_size])
    return chunks
 
 
def create_client():
    """Load the API key from the environment and return a Gemini client."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Check your .env file."
        )
    client = genai.Client(api_key=api_key)
    return client
 
 
def embed_text(client, text):
    """Embed a single text string and return its vector."""
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    return result.embeddings[0].values
 
 
def embed_all_chunks(client, chunks):
    """Embed every chunk in batches to respect the free tier rate limit."""
    BATCH_SIZE = 90
    embeddings = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        for chunk in batch:
            embeddings.append(embed_text(client, chunk))
        if i + BATCH_SIZE < len(chunks):
            print("Rate limit pause — waiting 60 seconds...")
            time.sleep(60)
    return embeddings
 
 
def cosine_similarity(vec_a, vec_b):
    """Return the cosine similarity between two vectors."""
    dot = np.dot(vec_a, vec_b)
    norm = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    return dot / norm
 
 
def search(client, query, chunks, embeddings, top_k=3):
    """Return the top_k most relevant chunks for the given query."""
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    query_vector = result.embeddings[0].values
    scores = [
        (cosine_similarity(query_vector, emb), chunk)
        for emb, chunk in zip(embeddings, chunks)
    ]
    scores.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scores[:top_k]]
 
 
def build_prompt(question, context_chunks):
    """Assemble a RAG prompt from the question and retrieved chunks."""
    context = "\n\n".join(context_chunks)
    prompt = (
        "You are a helpful assistant. Answer the question using only the "
        "context below.\n"
        'If the answer is not in the context, say "I don\'t know."\n\n'
        f"Context:\n{context}\n\n"
        f"Question:\n{question}"
    )
    return prompt
 
 

def generate_answer(prompt, max_retries=5):
    """Generate an answer with retries for temporary API errors."""

    for attempt in range(max_retries):
        try:
             client = Groq(api_key=os.getenv("GROQ_API_KEY"))
                        
             response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {"role": "user", "content": prompt}
                ],
             )
             return response.choices[0].message.content

        except Exception as e:
            error_message = str(e)

            # Retry only temporary server errors and rate limits.
            retryable = (
                "503" in error_message
                or "429" in error_message
                or "500" in error_message
                or "UNAVAILABLE" in error_message
                or "RESOURCE_EXHAUSTED" in error_message
                or "INTERNAL" in error_message
            )

            if not retryable or attempt == max_retries - 1:
                raise

            delay = min(2 ** attempt + random.random(), 30)

            print(
                f"Gemini is temporarily unavailable. "
                f"Retrying in {delay:.1f} seconds "
                f"(attempt {attempt + 1}/{max_retries})..."
            )

            time.sleep(delay)

 
 
def save_embeddings(chunks, embeddings, cache_path):
    """Save chunks and embeddings to a JSON file."""
    data = {"chunks": chunks, "embeddings": embeddings}
    with open(cache_path, "w") as f:
        json.dump(data, f)
 
 
def load_embeddings(cache_path):
    """Load chunks and embeddings from a JSON file."""
    if not os.path.exists(cache_path):
        return None
    with open(cache_path) as f:
        data = json.load(f)
    return data["chunks"], data["embeddings"]
 
 
def main():
    pdf_path = sys.argv[1]
    question = sys.argv[2]
    cache_path = pdf_path + ".cache.json"
    client = create_client()
 
    print(f"Loading {pdf_path}...")
    cached = load_embeddings(cache_path)
    if cached:
        chunks, embeddings = cached
        print(f"Loaded cache from {cache_path}")
    else:
        text = extract_text(pdf_path)
        chunks = chunk_text(text)
        print(f"No cache found. Embedding {len(chunks)} chunks...")
        embeddings = embed_all_chunks(client, chunks)
        save_embeddings(chunks, embeddings, cache_path)
        print(f"Cache saved to {cache_path}")
 
    top_chunks = search(client, question, chunks, embeddings)
    prompt = build_prompt(question, top_chunks)
    answer = generate_answer(prompt)
 
    print("\nAnswer:")
    print(answer)
 
 
if __name__ == "__main__":
    main()