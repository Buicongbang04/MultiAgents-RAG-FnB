from __future__ import annotations

import asyncio
import threading
from functools import lru_cache
from typing import List, Tuple

from app.core.logging import get_logger
from app.core.schemas import RetrievedSource

logger = get_logger(__name__)


class BGEReranker:
    """
    Cross-encoder reranker dùng BAAI/bge-reranker-v2-m3.
    Chỉ load model khi được gọi lần đầu (lazy init).
    """

    def __init__(self, model_name: str, device: str, dtype: str = "float32") -> None:
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self._model = None
        self._load_lock = threading.Lock()

    def _load(self):
        # Double-checked lock: arerank chạy trong nhiều thread, tránh nạp model hai lần (VRAM spike).
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            self._load_unlocked()

    def _load_unlocked(self):
        try:
            import torch
            from sentence_transformers import CrossEncoder

            # fp16 chỉ trên GPU; CPU ép fp32.
            use_fp16 = self.dtype == "float16" and self.device.startswith("cuda")
            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
                model_kwargs={"dtype": torch.float16 if use_fp16 else torch.float32},
            )
            logger.info("BGE reranker loaded: %s on %s", self.model_name, self.device)
        except Exception as exc:
            logger.warning("BGE reranker load failed (%s), reranker disabled.", exc)
            self._model = None

    def rerank(
        self,
        query: str,
        sources: List[RetrievedSource],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[RetrievedSource]:
        """
        Rerank sources bằng CrossEncoder. Trả về top_k sources theo score mới.
        Fallback về original order nếu model không load được.
        """
        if not sources:
            return sources

        self._load()

        if self._model is None:
            return sources[:top_k]

        pairs: List[Tuple[str, str]] = [(query, src.text) for src in sources]

        try:
            scores = self._model.predict(pairs)
        except Exception as exc:
            logger.warning("Reranker predict failed: %s", exc)
            return sources[:top_k]

        ranked = sorted(
            zip(sources, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results = []
        for src, score in ranked[:top_k]:
            if score < threshold:
                continue
            src.metadata["reranker_score"] = float(score)
            src.score = float(score)
            results.append(src)

        return results

    async def arerank(
        self,
        query: str,
        sources: List[RetrievedSource],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[RetrievedSource]:
        """Như rerank() nhưng chạy ở thread riêng để không chặn event loop."""
        return await asyncio.to_thread(self.rerank, query, sources, top_k, threshold)

    def warmup(self) -> None:
        """Nạp model và chạy một cặp để CUDA kernel được khởi tạo."""
        self._load()
        if self._model is not None:
            self._model.predict([("khởi động", "khởi động")])


class NullReranker:
    """No-op reranker — trả về sources[:top_k] không thay đổi score."""

    def rerank(
        self,
        query: str,
        sources: List[RetrievedSource],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[RetrievedSource]:
        return sources[:top_k]

    async def arerank(
        self,
        query: str,
        sources: List[RetrievedSource],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[RetrievedSource]:
        return sources[:top_k]

    def warmup(self) -> None:
        return None


@lru_cache(maxsize=1)
def get_reranker():
    from app.core.config import get_settings
    settings = get_settings()

    config = settings.reranker
    if config.backend == "bge":
        return BGEReranker(model_name=config.model, device=config.device, dtype=config.dtype)

    return NullReranker()
