"""Unit tests for the BGE reranker sigmoid + thresholding fix."""
from app.core.constants import SourceType
from app.core.schemas import RetrievedSource
from app.rag.reranker import BGEReranker, _sigmoid


def test_sigmoid_bounds_and_midpoint():
    assert _sigmoid(0.0) == 0.5
    assert _sigmoid(100.0) > 0.99
    assert _sigmoid(-100.0) < 0.01


def test_sigmoid_monotonic():
    assert _sigmoid(-2.0) < _sigmoid(0.0) < _sigmoid(2.0)


class _FakeCrossEncoder:
    """Returns fixed logits: strong-positive for the first pair, strong-negative for the rest."""

    def __init__(self, logits):
        self._logits = logits

    def predict(self, pairs):
        return self._logits[: len(pairs)]


def _src(source_id: str, score: float) -> RetrievedSource:
    return RetrievedSource(
        source_id=source_id,
        source_type=SourceType.FAQ,
        text=f"text-{source_id}",
        score=score,
    )


def test_rerank_applies_sigmoid_and_threshold():
    reranker = BGEReranker()
    reranker._model = _FakeCrossEncoder([10.0, -10.0])  # inject fake model, skip real load

    sources = [_src("a", 0.3), _src("b", 0.3)]
    out = reranker.rerank("q", sources, top_k=5, threshold=0.5)

    # logit -10 → prob ~0 is dropped by the 0.5 threshold; logit 10 → prob ~1 survives.
    assert [s.source_id for s in out] == ["a"]
    assert 0.99 < out[0].score <= 1.0
    # upstream score preserved for debugging
    assert out[0].metadata["pre_rerank_score"] == 0.3
    assert 0.99 < out[0].metadata["reranker_score"] <= 1.0
