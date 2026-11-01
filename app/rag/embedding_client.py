from __future__ import annotations

import asyncio
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
    def __init__(self, dim: int) -> None:
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
        device: str,
        batch_size: int,
        dtype: str = "float32",
    ) -> None:
        # torch/sentence_transformers chỉ import khi thật sự nạp model (unit test mock không cần torch).
        import torch
        from sentence_transformers import SentenceTransformer

        # fp16 chỉ có nghĩa trên GPU; CPU luôn dùng fp32.
        torch_dtype = torch.float16 if (dtype == "float16" and device.startswith("cuda")) else torch.float32

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

        self.model = SentenceTransformer(
            model_name,
            device=device,
            model_kwargs={"dtype": torch_dtype},
        )

    async def embed_text(self, text: str) -> list[float]:
        # encode là blocking (GPU/CPU) → chạy ở thread riêng để không chặn event loop.
        embeddings = await asyncio.to_thread(
            self.model.encode,
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

        embeddings = await asyncio.to_thread(
            self.model.encode,
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

    config = settings.embedding

    if config.backend == "mock":
        return MockEmbeddingClient(dim=config.dim)

    if config.backend == "sentence_transformers":
        return SentenceTransformersEmbeddingClient(
            model_name=config.model,
            device=config.device,
            batch_size=config.batch_size,
            dtype=config.dtype,
        )

    raise ValueError(f"Unsupported embedding backend: {config.backend}")