from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np

from app.cache.exact_cache import normalize_cache_text
from app.core.config import get_lexicon
from app.core.constants import Intent


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
        ttl_seconds: int,
        max_size: int,
        thresholds: Dict[str, float],
        default_threshold: float,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.thresholds = dict(thresholds)
        self.default_threshold = default_threshold
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
        return self.thresholds.get(intent.value, self.default_threshold)

    @staticmethod
    def _extract_domain_tags(text: str) -> Set[str]:
        text = normalize_cache_text(text)
        domain_tags = get_lexicon().semantic_cache.domain_tags

        tags: Set[str] = set()

        for tag, patterns in domain_tags.items():
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
        # Chỉ coi là cùng câu hỏi khi bộ tag trùng khớp hoàn toàn. Chỉ cần giao nhau
        # (bản cũ) thì "gợi ý cà phê đậm" sẽ hit cache của "gợi ý cà phê ít ngọt".
        if not query_tags or query_tags != entry_tags:
            return False

        if intent.value == "faq":
            return query_tags <= set(get_lexicon().semantic_cache.faq_alias_tags)

        if intent.value == "consultant":
            # "recommendation" có mặt ở gần như mọi câu tư vấn → không đủ để phân biệt.
            # Tag coffee/tea/budget chưa mô tả hết khẩu vị (ít ngọt, giải nhiệt...),
            # nên consultant chỉ hit bằng embedding similarity, không qua alias.
            return False

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

        alias_hit = self._is_safe_domain_alias_hit(
            intent=intent,
            query_tags=query_tags,
            entry_tags=best_entry_tags,
        )

        if best_similarity < threshold and not alias_hit:
            print(
                "[SEMANTIC CACHE MISS]",
                "intent=", intent.value,
                "text=", text,
                "best_similarity=", round(best_similarity, 4),
                "threshold=", threshold,
                "query_tags=", sorted(query_tags),
                "entry_tags=", sorted(best_entry_tags),
                "best_match=", best_entry.key_text,
                "entries=", len(self._entries),
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