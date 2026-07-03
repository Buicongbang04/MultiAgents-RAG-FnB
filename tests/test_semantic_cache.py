"""Unit tests for the semantic-cache domain-alias similarity floor fix."""
from app.cache.semantic_cache import SemanticInMemoryCache
from app.core.constants import Intent


def test_alias_hit_requires_similarity_floor():
    cache = SemanticInMemoryCache(thresholds={"faq": 0.9}, alias_similarity_floor=0.75)
    cache.set(Intent.FAQ, "wifi quán tên gì", [1.0, 0.0, 0.0], {"answer": "SSID X"})

    # Shares the 'wifi' domain tag but is near-orthogonal (cosine ~0) → below the
    # alias floor → must be a MISS (previously served the wrong cached answer).
    got = cache.get(Intent.FAQ, "mật khẩu wifi là gì", [0.0, 1.0, 0.0])
    assert got is None


def test_alias_hit_allowed_above_floor_below_threshold():
    cache = SemanticInMemoryCache(thresholds={"faq": 0.99}, alias_similarity_floor=0.75)
    cache.set(Intent.FAQ, "wifi quán tên gì", [1.0, 0.0, 0.0], {"answer": "SSID X"})

    # cosine ~0.8: below the 0.99 threshold but above the 0.75 alias floor and shares
    # the wifi tag → domain-alias hit is allowed.
    got = cache.get(Intent.FAQ, "cho hỏi wifi", [0.8, 0.6, 0.0])
    assert got is not None
    assert got["value"]["answer"] == "SSID X"


def test_exact_embedding_hit():
    cache = SemanticInMemoryCache(thresholds={"faq": 0.9})
    cache.set(Intent.FAQ, "wifi quán tên gì", [1.0, 0.0, 0.0], {"answer": "SSID X"})

    got = cache.get(Intent.FAQ, "wifi quán tên gì?", [1.0, 0.0, 0.0])
    assert got is not None
    assert got["metadata"]["cache_type"] == "semantic"
