from __future__ import annotations

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

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cuda") -> None:
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, device=self.device)
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


@lru_cache(maxsize=1)
def get_reranker():
    from app.core.config import get_settings
    settings = get_settings()

    backend = getattr(settings, "reranker_backend", "null")
    if backend == "bge":
        model = getattr(settings, "reranker_model", "BAAI/bge-reranker-v2-m3")
        device = getattr(settings, "reranker_device", "cuda")
        return BGEReranker(model_name=model, device=device)

    return NullReranker()
