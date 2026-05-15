from __future__ import annotations

import hashlib
from typing import List

import numpy as np

from app.core.config import get_settings


class BaseEmbeddingClient:
    async def embed_text(self, text: str) -> list[float]:
        raise NotImplementedError

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed_text(text) for text in texts]


class MockEmbeddingClient(BaseEmbeddingClient):
    """
    Mock embedding client for testing and development.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    async def embed_text(self, text: str) -> list[float]:
        vector = np.zeros(self.dim, dtype=np.float32)

        tokens = text.lower().split()

        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            idx = int(digest, 16) % self.dim
            vector[idx] += 1.0

        norm = np.linalg.norm(vector)

        if norm > 0:
            vector = vector / norm

        return vector.tolist()


def get_embedding_client() -> BaseEmbeddingClient:
    settings = get_settings()

    backend = getattr(settings, "embedding_backend", "mock")

    if backend == "mock":
        return MockEmbeddingClient(
            dim=getattr(settings, "embedding_dim", 384)
        )

    raise ValueError(f"Unsupported embedding backend: {backend}")