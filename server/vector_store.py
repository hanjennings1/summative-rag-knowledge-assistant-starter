from typing import Any, List

import chromadb
from chromadb.config import Settings
import requests

from config import Config
from documents import DocumentChunk


def get_chroma_client():
    """Create and return a persistent Chroma client."""
    # Saves vectors to disk at CHROMA_PATH; telemetry off to keep logs clean.
    return chromadb.PersistentClient(
        path=Config.CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )


def get_or_create_collection():
    """Get or create the Chroma collection for the knowledge assistant."""
    # Reuses the collection if it already exists, so re-running is safe.
    client = get_chroma_client()
    return client.get_or_create_collection(name=Config.COLLECTION_NAME)


def get_embedding(text: str) -> list[float]:
    """Create an embedding for a piece of text using the local model service."""
    # Send the text to Ollama's embed endpoint, using the model set in .env.
    # Note: /api/embed expects "input" (the older /api/embeddings used "prompt").
    response = requests.post(
        f"{Config.OLLAMA_BASE_URL}/api/embed",
        json={
            "model": Config.EMBEDDING_MODEL,
            "input": text,
        },
        timeout=120,        # Avoid hanging forever if Ollama stalls.
    )
    # Raise a clear error if Ollama returns a bad status (ex: 'model not found').
    response.raise_for_status()
    # Ollama returns a list of embeddings (one per input); we sent one text, so return the first.
    return response.json()["embeddings"][0]


def seed_vector_store(chunks: List[DocumentChunk]) -> int:
    """Add document chunks to the Chroma collection."""
    # Get or create the collection.
    collection = get_or_create_collection()

    # Chroma takes parallel lists: item i in each list belongs to the same chunk.
    ids = []
    documents = []
    metadatas = []
    embeddings = []

    # Convert each chunk into: id, document text, metadata, and embedding.
    for chunk in chunks:
        ids.append(chunk.id)
        documents.append(chunk.text)
        # Keep source metadata because the frontend needs to display sources.
        metadatas.append(
            {
                "source": chunk.source,
                "title": chunk.title,
                "chunk_index": chunk.chunk_index,
            }
        )
        embeddings.append(get_embedding(chunk.text))
    
    # Add or update the chunks in Chroma (recommend using collection.upsert(...) to prevent duplicating existing records).
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    
    # Return the number of chunks added.
    return len(chunks)


def retrieve_relevant_chunks(question: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """Retrieve relevant chunks for a user question."""
    # Fall back to TOP_K from .env if no value is passed in.
    top_k = top_k or Config.TOP_K

    # Create an embedding for the question.
    question_embedding = get_embedding(question)

    # Query the Chroma collection.
    collection = get_or_create_collection()     # reconnects to the collection the user seeded
    results = collection.query(
        query_embeddings = [question_embedding], # Chroma accepts multiple questions at once, this is sent as list of one
        n_results = top_k,                       # uses top_k from .env file, asks for this number of chunks
        include = ["documents", "metadatas", "distances"],   # default fields, written for documentation sake
    )

    # Return a list of dictionaries with: text, source, title, chunk_index (opt: distance/score)
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    chunks = []

    # (Based on example provided in starter comments)
    # zip pairs up item i from each list, since they all describe the same chunk.
    for text, metadata, distance in zip(documents, metadatas, distances):
        chunks.append(
            {
                "text": text,
                "source": metadata.get("source", "unknown"),
                "title": metadata.get("title", "Unknown Source"),
                "chunk_index": metadata.get("chunk_index"),
                "distance": distance,
            }
        )

    return chunks
