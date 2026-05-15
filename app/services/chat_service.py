import time

from app.agents.dispatcher import agent_dispatcher
from app.agents.router_agent import router_agent
from app.cache.cache_service import cache_service
from app.core.logging import get_logger
from app.core.schemas import (
    AgentInput,
    ChatRequest,
    ChatResponse,
    RAGQuery,
    RouterInput,
)
from app.queueing.request_queue import queue_manager
from app.session.session_store import session_store
from app.cache.intent_extractor import (
    IntentExtractionInput,
    get_intent_extractor,
)
from app.core.config import get_settings

logger = get_logger(__name__)


class ChatService:
    """
    End-to-end orchestration service.

    Flow:
    request
    → session
    → router
    → exact cache
    → semantic cache
    → dispatcher
    → agent
    → save cache
    → save history
    → response
    """

    async def chat(self, request: ChatRequest) -> ChatResponse:
        start = time.perf_counter()

        session = await session_store.get_or_create(request.session_id)

        await session_store.add_user_message(
            session.session_id,
            request.text,
        )

        router_output = await queue_manager.router.run(
            router_agent.classify,
            RouterInput(
                session_id=session.session_id,
                text=request.text,
                history=session.recent_history(),
                language=request.language,
            ),
        )

        settings = get_settings()
        extraction = None
        cache_lookup_text = request.text

        if getattr(settings, "intent_extractor_enabled", True):
            try:
                extraction = await get_intent_extractor().extract(
                    IntentExtractionInput(
                        session_id=session.session_id,
                        text=request.text,
                        intent=router_output.action,
                        language=router_output.language,
                        history=session.recent_history(),
                    )
                )

                if extraction.cache_key.strip():
                    cache_lookup_text = extraction.cache_key.strip()

                logger.info(
                    "Intent extracted session=%s intent=%s cache_key=%s action=%s context=%s",
                    session.session_id,
                    router_output.action.value,
                    cache_lookup_text,
                    extraction.action,
                    extraction.context,
                )

            except Exception as exc:
                logger.warning(
                    "Intent extraction failed session=%s intent=%s error=%s",
                    session.session_id,
                    router_output.action.value,
                    exc,
                )
                extraction = None
                cache_lookup_text = request.text

        cached = cache_service.get_exact(
            router_output.action,
            cache_lookup_text,
        )

        if cached is not None:
            logger.info(
                "Exact cache hit session=%s intent=%s",
                session.session_id,
                router_output.action.value,
            )

        if cached is None:
            try:
                cached = await cache_service.get_semantic(
                    router_output.action,
                    cache_lookup_text,
                )

                if cached is not None:
                    logger.info(
                        "Semantic cache hit session=%s intent=%s similarity=%s matched=%s",
                        session.session_id,
                        router_output.action.value,
                        cached["metadata"].get("similarity"),
                        cached["metadata"].get("matched_query"),
                    )
                else:
                    logger.info(
                        "Cache miss session=%s intent=%s text=%s",
                        session.session_id,
                        router_output.action.value,
                        request.text,
                    )

            except Exception as exc:
                logger.warning(
                    "Semantic cache lookup failed session=%s intent=%s error=%s",
                    session.session_id,
                    router_output.action.value,
                    exc,
                )
                cached = None

        if cached is not None:
            if cached["metadata"].get("cache_type") == "exact":
                try:
                    await cache_service.backfill_semantic_from_cached_value(
                        router_output.action,
                        cache_lookup_text,
                        cached["value"],
                    )
                except Exception as exc:
                    logger.warning(
                        "Semantic cache backfill failed session=%s intent=%s error=%s",
                        session.session_id,
                        router_output.action.value,
                        exc,
                    )

            latency_ms = (time.perf_counter() - start) * 1000

            await session_store.add_assistant_message(
                session.session_id,
                cached["value"]["answer"],
            )

            response = cache_service.build_chat_response_from_hit(
                session_id=session.session_id,
                cached=cached,
                latency_ms=latency_ms,
                router_metadata=router_output.metadata,
                queue_stats=queue_manager.stats(),
                extraction=extraction.model_dump() if extraction else None,
            )

            response.metadata["extraction"] = (
                extraction.model_dump() if extraction else None
            )
            response.metadata["cache_lookup_text"] = cache_lookup_text

            return response

        agent = agent_dispatcher.get_agent(router_output.action)

        agent_output = await queue_manager.generator.run(
            agent.run,
            AgentInput(
                session_id=session.session_id,
                text=request.text,
                intent=router_output.action,
                history=session.recent_history(),
                language=router_output.language,
                metadata={
                    "rag_query": RAGQuery(
                        query=request.text,
                        intent=router_output.action,
                        language=router_output.language,
                    ),
                    "extraction": extraction.model_dump() if extraction else None,
                    "cache_lookup_text": cache_lookup_text,
                },
            ),
        )

        try:
            cache_metadata = await cache_service.save_from_agent_output(
                router_output.action,
                cache_lookup_text,
                agent_output,
            )

            logger.info(
                "Cache saved session=%s intent=%s stored=%s",
                session.session_id,
                router_output.action.value,
                cache_metadata.get("stored"),
            )

        except Exception as exc:
            logger.warning(
                "Cache save failed session=%s intent=%s error=%s",
                session.session_id,
                router_output.action.value,
                exc,
            )
            cache_metadata = {
                "enabled": cache_service.enabled,
                "hit": False,
                "stored": False,
                "error": str(exc),
            }

        await session_store.add_assistant_message(
            session.session_id,
            agent_output.answer,
        )

        latency_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "Chat done session=%s intent=%s latency=%.2fms",
            session.session_id,
            router_output.action.value,
            latency_ms,
        )

        return ChatResponse(
            session_id=session.session_id,
            intent=router_output.action,
            agent=agent_output.agent,
            answer=agent_output.answer,
            language=agent_output.language,
            sources=agent_output.sources,
            latency_ms=latency_ms,
            metadata={
                "router": router_output.metadata,
                "queue_stats": queue_manager.stats(),
                "cache": cache_metadata,
                "extraction": extraction.model_dump() if extraction else None,
                "cache_lookup_text": cache_lookup_text,
            }
        )


chat_service = ChatService()