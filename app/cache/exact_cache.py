import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from app.core.constants import Intent


def normalize_cache_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[?!。！？]+$", "", text)
    return text.strip()


def build_exact_cache_key(intent: Intent, text: str) -> str:
    return f"{intent.value}:{normalize_cache_text(text)}"


@dataclass
class ExactCacheEntry:
    key: str
    value: Dict[str, Any]
    created_at: float
    expires_at: float
    hit_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExactInMemoryCache:
    def __init__(self, ttl_seconds: int = 1800, max_size: int = 1000):
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._store: OrderedDict[str, ExactCacheEntry] = OrderedDict()

    def get(self, key: str) -> Optional[ExactCacheEntry]:
        now = time.time()
        entry = self._store.get(key)

        if entry is None:
            return None

        if entry.expires_at <= now:
            self._store.pop(key, None)
            return None

        entry.hit_count += 1
        self._store.move_to_end(key)
        return entry

    def set(
        self,
        key: str,
        value: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExactCacheEntry:
        now = time.time()
        entry = ExactCacheEntry(
            key=key,
            value=value,
            created_at=now,
            expires_at=now + self.ttl_seconds,
            metadata=metadata or {},
        )

        self._store[key] = entry
        self._store.move_to_end(key)

        while len(self._store) > self.max_size:
            self._store.popitem(last=False)

        return entry

    def stats(self) -> Dict[str, Any]:
        return {
            "size": len(self._store),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
        }