from typing import Any

import requests

from config import Config
from vector_store import retrieve_relevant_chunks


def answer_question(question: str) -> dict[str, Any]:
    """Run the RAG workflow for a user question: PROVIDED"""
    chunks = retrieve_relevant_chunks(question, top_k=Config.TOP_K)

    if not chunks:
        return {
            "answer": (
                "I could not find enough relevant information in the provided "
                "knowledge base to answer that question."
            ),
            "sources": [],
        }

    prompt = build_prompt(question, chunks)
    answer = call_generation_model(prompt)

    return {
        "answer": answer,
        "sources": format_sources(chunks),
    }


def build_prompt(question: str, chunks: list[dict[str, Any]]) -> str:
    """Build a prompt that asks the model to answer using only retrieved context: PROVIDED
    You may refine it if needed."""
    context_blocks = []

    for index, chunk in enumerate(chunks, start=1):
        title = chunk.get("title", "Unknown Source")
        source = chunk.get("source", "unknown")
        text = chunk.get("text", "")

        context_blocks.append(
            f"[Source {index}: {title} | {source}]\n{text}"
        )

    context = "\n\n".join(context_blocks)

    return f"""
You are a helpful internal knowledge assistant.

Use only the provided context to answer the user's question.
If the context does not contain enough information, say that you do not have enough information from the knowledge base.

Keep the answer clear and practical.
Do not invent policies, steps, or facts that are not supported by the context.

Context:
{context}

User question:
{question}

Answer:
""".strip()


def call_generation_model(prompt: str) -> str:
    """Send the final prompt to the configured generation model."""
    # - Send a POST request to the Ollama generation endpoint
    # - Use Config.OLLAMA_BASE_URL. /  Config.GENERATION_MODEL. / Config.TEMPERATURE.
    response = requests.post(
        f"{Config.OLLAMA_BASE_URL}/api/generate",
        json={
            "model": Config.GENERATION_MODEL,
            "prompt": prompt,
            "stream": False,      # Request a non-streaming response (one complete answer)
            "options": {
                "temperature": Config.TEMPERATURE,
            },
        },
        timeout=120,  # Generation is slower than embedding, especially on the first call.
    )

    # Raise a clear error if Ollama returns a bad status.
    response.raise_for_status()

    # Return the generated response text.
    return response.json()["response"].strip()


def format_sources(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Format retrieved chunks for the frontend: PROVIDED
    You may adjust the excerpt length if needed."""
    sources = []

    for chunk in chunks:
        text = chunk.get("text", "")
        excerpt = text[:280] + "..." if len(text) > 280 else text

        sources.append(
            {
                "title": chunk.get("title", "Unknown Source"),
                "source": chunk.get("source", "unknown"),
                "chunk_index": chunk.get("chunk_index"),
                "excerpt": excerpt,
            }
        )

    return sources
