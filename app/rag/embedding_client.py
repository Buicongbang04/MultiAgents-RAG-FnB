from __future__ import annotations

import hashlib
from functools import lru_cache

import numpy as np

from app.core.config import get_settings

class BaseEmbeddingClient:
    async def embed_text(self, text: str) -> list[float]:
        raise NotImplementedError

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed_text(text) for text in texts]

class MockEmbeddingClient(BaseEmbeddingClient):
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


class SentenceTransformersEmbeddingClient(BaseEmbeddingClient):
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        batch_size: int = 16,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

        self.model = SentenceTransformer(
            model_name,
            device=device,
        )

    async def embed_text(self, text: str) -> list[float]:
        embeddings = self.model.encode(
            [text],
            batch_size=1,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return embeddings[0].astype(np.float32).tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        return embeddings.astype(np.float32).tolist()



@lru_cache(maxsize=1)
def get_embedding_client() -> BaseEmbeddingClient:
    settings = get_settings()

    backend = getattr(settings, "embedding_backend", "mock")

    if backend == "mock":
        return MockEmbeddingClient(
            dim=getattr(settings, "embedding_dim", 384),
        )

    if backend == "sentence_transformers":
        return SentenceTransformersEmbeddingClient(
            model_name=getattr(settings, "embedding_model", "BAAI/bge-m3"),
            device=getattr(settings, "embedding_device", "cuda"),
            batch_size=getattr(settings, "embedding_batch_size", 16),
        )

    raise ValueError(f"Unsupported embedding backend: {backend}")