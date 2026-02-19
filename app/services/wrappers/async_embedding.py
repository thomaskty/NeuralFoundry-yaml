# app/services/wrappers/async_embedding.py
import numpy as np
from app.services.embedding.openai_embedding import get_embedding_client

# Use OpenAI for embeddings
_embedding_client = get_embedding_client()


async def get_embedding_async(text: str) -> np.ndarray:
    """
    Get embedding for a single text using OpenAI.
    Returns 1536-dimensional vector.
    """
    return await _embedding_client.get_embedding(text)


async def get_batch_embeddings_async(texts: list[str]) -> list[np.ndarray]:
    """
    Get embeddings for multiple texts using OpenAI batch API.
    More efficient than calling get_embedding_async in a loop.
    """
    return await _embedding_client.get_batch_embeddings(texts)
