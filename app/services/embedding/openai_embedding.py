# app/services/embedding/openai_embedding.py
import os
import numpy as np
from openai import AsyncOpenAI
from app.core.config import settings


class OpenAIEmbeddingClient:
    """
    OpenAI embedding client using text-embedding-3-small (1536 dimensions)
    Async implementation for better performance.
    """

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.EMBEDDING_MODEL  # "text-embedding-3-small"
        self.embedding_dim = settings.EMBEDDING_DIMENSIONS  # 1536

    async def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a single text.

        Args:
            text: Input text string

        Returns:
            numpy array of shape (1536,)
        """
        response = await self.client.embeddings.create(
            model=self.model,
            input=text
        )
        embedding = response.data[0].embedding
        return np.array(embedding, dtype=np.float32)

    async def get_batch_embeddings(self, texts: list[str]) -> list[np.ndarray]:
        """
        Get embeddings for multiple texts in one API call (more efficient).

        Args:
            texts: List of text strings

        Returns:
            List of numpy arrays, each of shape (1536,)
        """
        # OpenAI allows batch requests up to 2048 texts
        # If you have more, you'll need to chunk them
        response = await self.client.embeddings.create(
            model=self.model,
            input=texts
        )

        embeddings = [np.array(item.embedding, dtype=np.float32) for item in response.data]
        return embeddings


# For backward compatibility - single global instance
_embedding_client = None


def get_embedding_client() -> OpenAIEmbeddingClient:
    """Get or create singleton embedding client"""
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = OpenAIEmbeddingClient()
    return _embedding_client