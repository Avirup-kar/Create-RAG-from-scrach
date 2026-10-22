
import os
import sys
import time
import hashlib
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types
import random


# -----------------------------
# Configuration
# -----------------------------

CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-3.5-flash"
BATCH_SIZE = 90


# -----------------------------
# 1. Extract text from PDF
# -----------------------------

def extract_text(pdf_path):
    """Extract text from every page of a PDF."""
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)

    pages = [
        page.extract_text() or ""
        for page in reader.pages
    ]

    return "\n".join(pages)


# -----------------------------
# 2. Split text into chunks
# -----------------------------

def chunk_text(text, chunk_size=500, overlap=100):
    """Split text into overlapping chunks."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []

    step = chunk_size - overlap

    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size]

        if chunk.strip():
            chunks.append(chunk)

    return chunks


# -----------------------------
# 3. Initialize Gemini
# -----------------------------

def create_client():
    """Create Gemini API client."""
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Check your .env file."
        )

    return genai.Client(api_key=api_key)


# -----------------------------
# 4. Generate Gemini embeddings
# -----------------------------

def embed_text(client, text, task_type):
    """Generate an embedding using Gemini."""
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type=task_type
        ),
    )

    return result.embeddings[0].values


def embed_all_chunks(client, chunks):
    """Embed chunks in batches with a pause between batches."""
    embeddings = []

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]

        for chunk in batch:
            embedding = embed_text(
                client,
                chunk,
                "RETRIEVAL_DOCUMENT",
            )
            embeddings.append(embedding)

        print(
            f"Embedded {len(embeddings)} "
            f"of {len(chunks)} chunks"
        )

        if start + BATCH_SIZE < len(chunks):
            print("Waiting 60 seconds for the next batch...")
            time.sleep(60)

    return embeddings


# -----------------------------
# 5. Initialize ChromaDB
# -----------------------------

def get_collection(pdf_path):
    """
    Create or retrieve a persistent Chroma collection.

    Different PDF contents get different collection names.
    """
    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    # Fingerprint the PDF contents, not just its filename.
    digest = hashlib.sha256()

    with open(pdf_path, "rb") as pdf_file:
        for block in iter(
            lambda: pdf_file.read(1024 * 1024), b""
        ):
            digest.update(block)

    pdf_hash = digest.hexdigest()[:16]
    collection_name = f"pdf_{pdf_hash}"

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    return collection


# -----------------------------
# 6. Store chunks in ChromaDB
# -----------------------------

def index_pdf(client, collection, pdf_path):
    """Embed a PDF and store its chunks in ChromaDB."""
    text = extract_text(pdf_path)

    if not text.strip():
        raise ValueError(
            "No text found in this PDF. "
            "It may contain scanned pages requiring OCR."
        )

    chunks = chunk_text(text)

    print(f"Found {len(chunks)} chunks.")

    embeddings = embed_all_chunks(client, chunks)

    # Store each chunk, its embedding, and its metadata.
    for start in range(0, len(chunks), BATCH_SIZE):
        batch_chunks = chunks[start:start + BATCH_SIZE]
        batch_embeddings = embeddings[
            start:start + BATCH_SIZE
        ]

        ids = [
            f"chunk_{i}"
            for i in range(start, start + len(batch_chunks))
        ]

        metadatas = [
            {
                "source": os.path.basename(pdf_path),
                "chunk_index": start + i,
            }
            for i in range(len(batch_chunks))
        ]

        collection.upsert(
            ids=ids,
            documents=batch_chunks,
            embeddings=batch_embeddings,
            metadatas=metadatas,
        )

    print(
        f"Stored {collection.count()} chunks in ChromaDB."
    )


# -----------------------------
# 7. Semantic search
# -----------------------------

def search(client, collection, query, top_k=3):
    """Retrieve the most relevant chunks from ChromaDB."""
    total = collection.count()

    if total == 0:
        return []

    query_embedding = embed_text(
        client,
        query,
        "RETRIEVAL_QUERY",
    )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, total),
        include=["documents", "distances"],
    )

    documents = results["documents"][0]

    print("\nRetrieved chunks:")

    for i, document in enumerate(documents):
        distance = results["distances"][0][i]
        print(f"\nChunk {i + 1} (distance: {distance:.4f})")
        print(document[:200], "...")

    return documents


# -----------------------------
# 8. Build RAG prompt
# -----------------------------

def build_prompt(question, context_chunks):
    """Build the prompt from retrieved context."""
    context = "\n\n".join(context_chunks)

    return (
        "You are a helpful assistant. Answer the question "
        "using only the context below.\n"
        'If the answer is not in the context, say "I don\'t know."\n\n'
        f"Context:\n{context}\n\n"
        f"Question:\n{question}"
    )


# -----------------------------
# 9. Generate answer
# -----------------------------


def generate_answer(client, prompt, max_retries=5):
    """Generate an answer with retries for temporary API errors."""

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=prompt,
            )
            return response.text or "No answer was generated."

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



# -----------------------------
# 10. Main
# -----------------------------

def main():
    if len(sys.argv) < 3:
        print(
            'Usage: python main.py "documents/python.pdf" '
            '"What is Python?"'
        )
        return

    pdf_path = str(Path(sys.argv[1]).resolve())
    question = " ".join(sys.argv[2:])

    if not os.path.isfile(pdf_path):
        print(f"PDF not found: {pdf_path}")
        return

    client = create_client()
    collection = get_collection(pdf_path)

    # Only embed the PDF if it is not already indexed.
    if collection.count() == 0:
        print(f"Indexing PDF: {pdf_path}")
        index_pdf(client, collection, pdf_path)
    else:
        print(
            f"Found {collection.count()} existing chunks "
            "in ChromaDB. Skipping indexing."
        )

    top_chunks = search(
        client,
        collection,
        question,
        top_k=3,
    )

    if not top_chunks:
        print("No relevant context found.")
        return

    prompt = build_prompt(question, top_chunks)
    answer = generate_answer(client, prompt)

    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()
