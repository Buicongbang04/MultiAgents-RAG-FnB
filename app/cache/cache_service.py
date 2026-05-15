from typing import Any, Dict, Optional, Set

from app.cache.exact_cache import ExactInMemoryCache, build_exact_cache_key
from app.cache.semantic_cache import SemanticInMemoryCache
from app.core.config import get_settings
from app.core.constants import Intent
from app.core.schemas import AgentOutput, ChatResponse
from app.rag.embedding_client import get_embedding_client


class CacheService:
    def __init__(self):
        settings = get_settings()

        self.enabled = getattr(settings, "cache_enabled", True)
        self.exact_enabled = getattr(settings, "exact_cache_enabled", True)
        self.semantic_enabled = getattr(settings, "semantic_cache_enabled", True)

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

        self.semantic_cache = SemanticInMemoryCache(
            ttl_seconds=getattr(settings, "semantic_cache_ttl_seconds", 1800),
            max_size=getattr(settings, "semantic_cache_max_size", 1000),
            thresholds={
                "faq": getattr(settings, "semantic_cache_faq_threshold", 0.95),
                "consultant": getattr(settings, "semantic_cache_consultant_threshold", 0.94),
                "ignore": getattr(settings, "semantic_cache_ignore_threshold", 0.97),
            },
        )

    @staticmethod
    def _parse_csv_setting(value: str) -> Set[str]:
        return {item.strip() for item in value.split(",") if item.strip()}

    def is_cacheable_intent(self, intent: Intent) -> bool:
        if not self.enabled:
            return False

        if intent.value in self.skip_intents:
            return False

        return intent.value in self.cacheable_intents

    def get_exact(self, intent: Intent, text: str) -> Optional[Dict[str, Any]]:
        if not self.exact_enabled or not self.is_cacheable_intent(intent):
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

    async def get_semantic(self, intent: Intent, text: str) -> Optional[Dict[str, Any]]:
        if not self.semantic_enabled or not self.is_cacheable_intent(intent):
            return None

        embedding = await get_embedding_client().embed_text(text)
        return self.semantic_cache.get(
            intent=intent,
            text=text,
            embedding=embedding,
        )

    def _value_from_agent_output(self, agent_output: AgentOutput) -> Dict[str, Any]:
        return {
            "session_id": agent_output.session_id,
            "intent": agent_output.intent,
            "agent": agent_output.agent,
            "answer": agent_output.answer,
            "language": agent_output.language,
            "sources": agent_output.sources,
            "metadata": agent_output.metadata,
        }

    def set_exact_from_agent_output(
        self,
        intent: Intent,
        text: str,
        agent_output: AgentOutput,
    ) -> Optional[Dict[str, Any]]:
        if not self.exact_enabled or not self.is_cacheable_intent(intent):
            return None

        if not agent_output.answer.strip():
            return None

        key = build_exact_cache_key(intent, text)
        value = self._value_from_agent_output(agent_output)

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

    async def set_semantic_from_agent_output(
        self,
        intent: Intent,
        text: str,
        agent_output: AgentOutput,
    ) -> Optional[Dict[str, Any]]:
        if not self.semantic_enabled or not self.is_cacheable_intent(intent):
            return None

        if not agent_output.answer.strip():
            return None

        embedding = await get_embedding_client().embed_text(text)
        value = self._value_from_agent_output(agent_output)

        entry = self.semantic_cache.set(
            intent=intent,
            text=text,
            embedding=embedding,
            value=value,
            metadata={
                "intent": intent.value,
                "agent": agent_output.agent.value,
            },
        )

        return {
            "enabled": True,
            "hit": False,
            "cache_type": "semantic",
            "stored": True,
            "matched_query": entry.key_text,
            "hit_count": entry.hit_count,
            "stats": self.semantic_cache.stats(),
        }

    async def save_from_agent_output(
        self,
        intent: Intent,
        text: str,
        agent_output: AgentOutput,
    ) -> Dict[str, Any]:
        exact_metadata = self.set_exact_from_agent_output(
            intent=intent,
            text=text,
            agent_output=agent_output,
        )

        semantic_metadata = await self.set_semantic_from_agent_output(
            intent=intent,
            text=text,
            agent_output=agent_output,
        )

        return {
            "enabled": self.enabled,
            "hit": False,
            "stored": bool(exact_metadata or semantic_metadata),
            "exact": exact_metadata,
            "semantic": semantic_metadata,
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
    
    async def backfill_semantic_from_cached_value(
            self,
            intent: Intent,
            text: str,
            cached_value: Dict[str, Any],
        ) -> Optional[Dict[str, Any]]:
            if not self.semantic_enabled or not self.is_cacheable_intent(intent):
                return None

            if not cached_value.get("answer", "").strip():
                return None

            embedding = await get_embedding_client().embed_text(text)

            entry = self.semantic_cache.set(
                intent=intent,
                text=text,
                embedding=embedding,
                value=cached_value,
                metadata={
                    "intent": intent.value,
                    "agent": getattr(cached_value.get("agent"), "value", str(cached_value.get("agent"))),
                    "source": "exact_cache_backfill",
                },
            )

            return {
                "enabled": True,
                "hit": False,
                "cache_type": "semantic",
                "stored": True,
                "source": "exact_cache_backfill",
                "matched_query": entry.key_text,
                "stats": self.semantic_cache.stats(),
            }
    
    def clear_all(self) -> Dict[str, Any]:
        self.exact_cache._store.clear()
        self.semantic_cache._entries.clear()

        return {
            "cleared": True,
            "exact": self.exact_cache.stats(),
            "semantic": self.semantic_cache.stats(),
        }

    def stats(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "exact_enabled": self.exact_enabled,
            "semantic_enabled": self.semantic_enabled,
            "cacheable_intents": sorted(self.cacheable_intents),
            "skip_intents": sorted(self.skip_intents),
            "exact": self.exact_cache.stats(),
            "semantic": self.semantic_cache.stats(),
        }


cache_service = CacheService()