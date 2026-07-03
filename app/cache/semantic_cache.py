from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np

from app.cache.exact_cache import normalize_cache_text
from app.core.constants import Intent
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SemanticCacheEntry:
    key_text: str
    normalized_text: str
    intent: Intent
    embedding: np.ndarray
    value: Dict[str, Any]
    created_at: float
    expires_at: float
    hit_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SemanticInMemoryCache:
    def __init__(
        self,
        ttl_seconds: int = 1800,
        max_size: int = 1000,
        thresholds: Optional[Dict[str, float]] = None,
        alias_similarity_floor: float = 0.75,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.thresholds = thresholds or {
            "faq": 0.85,
            "consultant": 0.82,
            "ignore": 0.90,
        }
        # A domain-alias match may relax the per-intent threshold, but never below
        # this cosine floor — otherwise a shared tag (e.g. "wifi") would serve a
        # cached answer even at ~0 similarity, returning the wrong response.
        self.alias_similarity_floor = alias_similarity_floor
        self._entries: List[SemanticCacheEntry] = []

    def _cleanup_expired(self) -> None:
        now = time.time()
        self._entries = [
            entry for entry in self._entries
            if entry.expires_at > now
        ]

    @staticmethod
    def _to_vector(embedding: List[float]) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def get_threshold(self, intent: Intent) -> float:
        return self.thresholds.get(intent.value, 0.85)

    @staticmethod
    def _extract_domain_tags(text: str) -> Set[str]:
        text = normalize_cache_text(text)

        tags: Set[str] = set()

        faq_patterns = {
            "wifi": [
                r"\bwifi\b",
                r"wi fi",
                r"internet",
                r"mạng",
                r"pass",
                r"password",
                r"mật khẩu",
            ],
            "opening_hours": [
                r"mấy giờ",
                r"giờ mở cửa",
                r"giờ đóng cửa",
                r"đóng cửa",
                r"mở cửa",
                r"open",
                r"close",
                r"closing",
            ],
            "payment": [
                r"thanh toán",
                r"momo",
                r"chuyển khoản",
                r"tiền mặt",
                r"visa",
                r"bank",
                r"payment",
                r"pay",
            ],
            "delivery": [
                r"giao hàng",
                r"ship",
                r"delivery",
                r"mang đi",
                r"take away",
                r"takeaway",
            ],
        }

        consultant_patterns = {
            "budget": [
                r"rẻ",
                r"giá mềm",
                r"tiết kiệm",
                r"ngon rẻ",
                r"không quá mắc",
                r"budget",
                r"cheap",
                r"affordable",
            ],
            "recommendation": [
                r"gợi ý",
                r"recommend",
                r"tư vấn",
                r"món nào",
                r"có gì ngon",
                r"nên uống",
                r"dễ uống",
                r"best",
            ],
            "coffee": [
                r"cà phê",
                r"coffee",
                r"bạc xỉu",
                r"latte",
                r"americano",
            ],
            "tea": [
                r"trà",
                r"tea",
                r"sen",
                r"đào",
                r"vải",
            ],
        }

        for tag, patterns in {**faq_patterns, **consultant_patterns}.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    tags.add(tag)
                    break

        return tags

    @staticmethod
    def _is_safe_domain_alias_hit(
        intent: Intent,
        query_tags: Set[str],
        entry_tags: Set[str],
    ) -> bool:
        overlap = query_tags & entry_tags

        if not overlap:
            return False

        if intent.value == "faq":
            safe_faq_tags = {
                "wifi",
                "opening_hours",
                "payment",
                "delivery",
            }
            return bool(overlap & safe_faq_tags)

        if intent.value == "consultant":
            safe_consultant_tags = {
                "budget",
                "recommendation",
                "coffee",
                "tea",
            }
            return bool(overlap & safe_consultant_tags)

        if intent.value == "ignore":
            return False

        return False

    def get(
        self,
        intent: Intent,
        text: str,
        embedding: List[float],
    ) -> Optional[Dict[str, Any]]:
        self._cleanup_expired()

        if not self._entries:
            return None

        query_vector = self._to_vector(embedding)
        threshold = self.get_threshold(intent)
        query_tags = self._extract_domain_tags(text)

        best_entry: Optional[SemanticCacheEntry] = None
        best_similarity = -1.0
        best_entry_tags: Set[str] = set()

        for entry in self._entries:
            if entry.intent != intent:
                continue

            similarity = self._cosine_similarity(query_vector, entry.embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_entry = entry
                best_entry_tags = self._extract_domain_tags(entry.key_text)

        if best_entry is None:
            return None

        alias_hit = (
            best_similarity >= self.alias_similarity_floor
            and self._is_safe_domain_alias_hit(
                intent=intent,
                query_tags=query_tags,
                entry_tags=best_entry_tags,
            )
        )

        if best_similarity < threshold and not alias_hit:
            logger.debug(
                "semantic cache miss intent=%s best_similarity=%.4f threshold=%.2f "
                "alias_floor=%.2f query_tags=%s entry_tags=%s best_match=%r entries=%d",
                intent.value,
                best_similarity,
                threshold,
                self.alias_similarity_floor,
                sorted(query_tags),
                sorted(best_entry_tags),
                best_entry.key_text,
                len(self._entries),
            )
            return None

        best_entry.hit_count += 1

        return {
            "entry": best_entry,
            "value": best_entry.value,
            "metadata": {
                "enabled": True,
                "hit": True,
                "cache_type": "semantic",
                "similarity": round(best_similarity, 4),
                "threshold": threshold,
                "matched_query": best_entry.key_text,
                "match_reason": "domain_alias" if alias_hit and best_similarity < threshold else "embedding",
                "query_tags": sorted(query_tags),
                "matched_tags": sorted(best_entry_tags),
                "hit_count": best_entry.hit_count,
                "stats": self.stats(),
            },
        }

    def set(
        self,
        intent: Intent,
        text: str,
        embedding: List[float],
        value: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SemanticCacheEntry:
        self._cleanup_expired()

        normalized_text = normalize_cache_text(text)
        vector = self._to_vector(embedding)
        now = time.time()

        for index, entry in enumerate(self._entries):
            if entry.intent == intent and entry.normalized_text == normalized_text:
                updated = SemanticCacheEntry(
                    key_text=text,
                    normalized_text=normalized_text,
                    intent=intent,
                    embedding=vector,
                    value=value,
                    created_at=now,
                    expires_at=now + self.ttl_seconds,
                    hit_count=entry.hit_count,
                    metadata=metadata or {},
                )
                self._entries[index] = updated
                return updated

        entry = SemanticCacheEntry(
            key_text=text,
            normalized_text=normalized_text,
            intent=intent,
            embedding=vector,
            value=value,
            created_at=now,
            expires_at=now + self.ttl_seconds,
            metadata=metadata or {},
        )

        self._entries.append(entry)

        while len(self._entries) > self.max_size:
            self._entries.pop(0)

        return entry

    def stats(self) -> Dict[str, Any]:
        return {
            "size": len(self._entries),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "thresholds": self.thresholds,
        }