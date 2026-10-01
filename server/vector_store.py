from typing import Any, List

import chromadb
from chromadb.config import Settings
import requests

from config import Config
from documents import DocumentChunk


def get_chroma_client():
    """Create and return a persistent Chroma client."""
    return chromadb.PersistentClient(
        path=Config.CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )


def get_or_create_collection():
    """Get or create the Chroma collection for the knowledge assistant."""
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
    """
    Add document chunks to the Chroma collection.

    TODO:
    - Get or create the collection.
    - Convert each chunk into:
        - id
        - document text
        - metadata with source, title, and chunk_index
        - embedding
    - Add or update the chunks in Chroma (recommend using collection.upsert(...) to prevent duplicating existing records).
    - Return the number of chunks added.

    Keep source metadata because the frontend needs to display sources.
    """
    raise NotImplementedError("TODO: Seed Chroma with document chunks and metadata.")


def retrieve_relevant_chunks(question: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """
    Retrieve relevant chunks for a user question.

    TODO:
    - Create an embedding for the question.
    - Query the Chroma collection.
    - Return a list of dictionaries with:
        - text
        - source
        - title
        - chunk_index
        - optional distance or score

    The RAG workflow expects a list shaped like this:

        [
            {
                "text": "Relevant source text...",
                "source": "product_support.txt",
                "title": "Product Support Guide",
                "chunk_index": 0
            }
        ]
    """
    raise NotImplementedError("TODO: Retrieve relevant chunks for the user question.")
