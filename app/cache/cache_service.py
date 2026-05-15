from typing import Any, Dict, Optional, Set

from app.cache.exact_cache import ExactInMemoryCache, build_exact_cache_key
from app.core.config import get_settings
from app.core.constants import Intent
from app.core.schemas import AgentOutput, ChatResponse


class CacheService:
    def __init__(self):
        settings = get_settings()

        self.enabled = getattr(settings, "cache_enabled", True)
        self.exact_enabled = getattr(settings, "exact_cache_enabled", True)

        self.cacheable_intents: Set[str] = self._parse_csv_setting(
            getattr(settings, "cacheable_intents", "faq,consultant,ignore")
        )
        self.skip_intents: Set[str] = self._parse_csv_setting(
            getattr(settings, "cache_skip_intents", "order")
        )

        self.exact_cache = ExactInMemoryCache(
            ttl_seconds=getattr(settings, "exact_cache_ttl_seconds", 1800),
            max_size=getattr(settings, "exact_cache_max_size", 1000),
        )

    @staticmethod
    def _parse_csv_setting(value: str) -> Set[str]:
        return {item.strip() for item in value.split(",") if item.strip()}

    def is_cacheable_intent(self, intent: Intent) -> bool:
        if not self.enabled or not self.exact_enabled:
            return False

        if intent.value in self.skip_intents:
            return False

        return intent.value in self.cacheable_intents

    def get_exact(self, intent: Intent, text: str) -> Optional[Dict[str, Any]]:
        if not self.is_cacheable_intent(intent):
            return None

        key = build_exact_cache_key(intent, text)
        entry = self.exact_cache.get(key)

        if entry is None:
            return None

        return {
            "key": key,
            "entry": entry,
            "value": entry.value,
            "metadata": {
                "enabled": True,
                "hit": True,
                "cache_type": "exact",
                "key": key,
                "hit_count": entry.hit_count,
                "stats": self.exact_cache.stats(),
            },
        }

    def set_exact_from_agent_output(
        self,
        intent: Intent,
        text: str,
        agent_output: AgentOutput,
    ) -> Optional[Dict[str, Any]]:
        if not self.is_cacheable_intent(intent):
            return None

        if not agent_output.answer.strip():
            return None

        key = build_exact_cache_key(intent, text)

        value = {
            "session_id": agent_output.session_id,
            "intent": agent_output.intent,
            "agent": agent_output.agent,
            "answer": agent_output.answer,
            "language": agent_output.language,
            "sources": agent_output.sources,
            "metadata": agent_output.metadata,
        }

        entry = self.exact_cache.set(
            key=key,
            value=value,
            metadata={
                "intent": intent.value,
                "agent": agent_output.agent.value,
            },
        )

        return {
            "enabled": True,
            "hit": False,
            "cache_type": "exact",
            "key": key,
            "stored": True,
            "hit_count": entry.hit_count,
            "stats": self.exact_cache.stats(),
        }

    def build_chat_response_from_hit(
        self,
        session_id: str,
        cached: Dict[str, Any],
        latency_ms: float,
        router_metadata: Dict[str, Any],
        queue_stats: Dict[str, Any],
    ) -> ChatResponse:
        value = cached["value"]

        return ChatResponse(
            session_id=session_id,
            intent=value["intent"],
            agent=value["agent"],
            answer=value["answer"],
            language=value["language"],
            sources=value["sources"],
            latency_ms=latency_ms,
            metadata={
                "router": router_metadata,
                "queue_stats": queue_stats,
                "cache": cached["metadata"],
            },
        )


cache_service = CacheService()