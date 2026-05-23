import time
from typing import AsyncGenerator

from app.agents.dispatcher import agent_dispatcher
from app.agents.router_agent import router_agent
from app.cache.cache_service import cache_service
from app.cache.intent_extractor import IntentExtractionInput, get_intent_extractor
from app.core.config import get_settings
from app.core.constants import AgentName, Intent, Language, StreamEventType
from app.core.logging import get_logger
from app.core.schemas import (
    AgentInput,
    AgentOutput,
    ChatRequest,
    ChatResponse,
    RAGQuery,
    RouterInput,
)
from app.llm import get_llm_client
from app.queueing.request_queue import queue_manager
from app.session.session_store import session_store
from app.streaming.sse import sse_event
from app.streaming.clause_splitter import ClauseSplitter

logger = get_logger(__name__)


class ChatService:
    """
    End-to-end orchestration.

    Flow (non-streaming):
      request → session → router → intent extractor → exact cache
      → semantic cache → agent.run() → save cache → save history → response

    Flow (streaming):
      request → session → router → intent extractor → exact/semantic cache
      → if hit: stream cached answer
      → if miss: agent.prepare() → llm.stream_generate() → pipe SSE
      → save cache + history after stream completes
    """

    async def _resolve_cache_lookup(
        self,
        request: ChatRequest,
        session_id: str,
        router_output,
    ):
        """Run intent extractor + cache lookup. Returns (extraction, cache_lookup_text, cached)."""
        settings = get_settings()
        extraction = None
        cache_lookup_text = request.text

        if getattr(settings, "intent_extractor_enabled", True):
            try:
                extraction = await get_intent_extractor().extract(
                    IntentExtractionInput(
                        session_id=session_id,
                        text=request.text,
                        intent=router_output.action,
                        language=router_output.language,
                        history=(await session_store.get(session_id)).recent_history()
                        if await session_store.get(session_id)
                        else [],
                    )
                )
                if extraction.cache_key.strip():
                    cache_lookup_text = extraction.cache_key.strip()
            except Exception as exc:
                logger.warning("Intent extraction failed session=%s error=%s", session_id, exc)

        cached = cache_service.get_exact(router_output.action, cache_lookup_text)

        if cached is None:
            try:
                cached = await cache_service.get_semantic(router_output.action, cache_lookup_text)
            except Exception as exc:
                logger.warning("Semantic cache lookup failed session=%s error=%s", session_id, exc)

        return extraction, cache_lookup_text, cached

    async def chat(self, request: ChatRequest) -> ChatResponse:
        start = time.perf_counter()

        session = await session_store.get_or_create(request.session_id)
        await session_store.add_user_message(session.session_id, request.text)

        router_output = await queue_manager.router.run(
            router_agent.classify,
            RouterInput(
                session_id=session.session_id,
                text=request.text,
                history=session.recent_history(),
                language=request.language,
            ),
        )

        extraction, cache_lookup_text, cached = await self._resolve_cache_lookup(
            request, session.session_id, router_output
        )

        if cached is not None:
            if cached["metadata"].get("cache_type") == "exact":
                try:
                    await cache_service.backfill_semantic_from_cached_value(
                        router_output.action, cache_lookup_text, cached["value"]
                    )
                except Exception as exc:
                    logger.warning("Semantic backfill failed: %s", exc)

            await session_store.add_assistant_message(
                session.session_id, cached["value"]["answer"]
            )
            latency_ms = (time.perf_counter() - start) * 1000
            response = cache_service.build_chat_response_from_hit(
                session_id=session.session_id,
                cached=cached,
                latency_ms=latency_ms,
                router_metadata=router_output.metadata,
                queue_stats=queue_manager.stats(),
                extraction=extraction.model_dump() if extraction else None,
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
                router_output.action, cache_lookup_text, agent_output
            )
        except Exception as exc:
            logger.warning("Cache save failed: %s", exc)
            cache_metadata = {"enabled": False, "hit": False, "stored": False}

        await session_store.add_assistant_message(session.session_id, agent_output.answer)
        latency_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "Chat done session=%s intent=%s latency=%.2fms",
            session.session_id, router_output.action.value, latency_ms,
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
            },
        )

    async def chat_stream(
        self,
        request: ChatRequest,
        start_time: float,
    ) -> AsyncGenerator[str, None]:
        """
        True streaming pipeline:
          router → cache → agent.prepare() → llm.stream_generate() → SSE tokens
        Saves to cache + session after stream finishes.
        """
        session = await session_store.get_or_create(request.session_id)
        await session_store.add_user_message(session.session_id, request.text)

        try:
            router_output = await queue_manager.router.run(
                router_agent.classify,
                RouterInput(
                    session_id=session.session_id,
                    text=request.text,
                    history=session.recent_history(),
                    language=request.language,
                ),
            )
        except Exception as exc:
            yield sse_event(StreamEventType.ERROR, {"message": str(exc)})
            yield "data: [DONE]\n\n"
            return

        extraction, cache_lookup_text, cached = await self._resolve_cache_lookup(
            request, session.session_id, router_output
        )

        # ── Cache hit: stream cached answer ──────────────────────────────────
        if cached is not None:
            answer = cached["value"]["answer"]
            ttft_ms = round((time.perf_counter() - start_time) * 1000, 2)
            yield sse_event(
                StreamEventType.METADATA,
                {
                    "ttft_ms": ttft_ms,
                    "session_id": session.session_id,
                    "intent": router_output.action.value,
                    "cache_hit": True,
                    "cache_type": cached["metadata"].get("cache_type"),
                },
            )

            splitter = ClauseSplitter()
            for word in answer.split():
                token = word + " "
                yield sse_event(StreamEventType.TOKEN, {"token": token})
                for clause in splitter.push(token):
                    yield sse_event(StreamEventType.CLAUSE, {"clause": clause})

            for clause in splitter.flush():
                yield sse_event(StreamEventType.CLAUSE, {"clause": clause})

            yield "data: [DONE]\n\n"

            await session_store.add_assistant_message(session.session_id, answer)
            return

        # ── Cache miss: prepare agent + stream LLM ───────────────────────────
        agent = agent_dispatcher.get_agent(router_output.action)

        try:
            agent_input = AgentInput(
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
                },
            )
            prepared = await agent.prepare(agent_input)
        except Exception as exc:
            logger.exception("Agent prepare failed session=%s", session.session_id)
            yield sse_event(StreamEventType.ERROR, {"message": str(exc)})
            yield "data: [DONE]\n\n"
            return

        llm = get_llm_client()
        splitter = ClauseSplitter()
        full_text = ""
        first_token = True

        try:
            async for token in llm.stream_generate(prepared.llm_request):
                if first_token:
                    ttft_ms = round((time.perf_counter() - start_time) * 1000, 2)
                    yield sse_event(
                        StreamEventType.METADATA,
                        {
                            "ttft_ms": ttft_ms,
                            "session_id": session.session_id,
                            "intent": router_output.action.value,
                            "cache_hit": False,
                        },
                    )
                    first_token = False

                full_text += token
                yield sse_event(StreamEventType.TOKEN, {"token": token})

                for clause in splitter.push(token):
                    yield sse_event(StreamEventType.CLAUSE, {"clause": clause})

        except Exception as exc:
            logger.exception("LLM stream failed session=%s", session.session_id)
            if not full_text:
                full_text = prepared.fallback_answer
            yield sse_event(StreamEventType.ERROR, {"message": "Stream error, using fallback"})

        for clause in splitter.flush():
            yield sse_event(StreamEventType.CLAUSE, {"clause": clause})

        yield "data: [DONE]\n\n"

        # ── Post-stream: save cache + history ────────────────────────────────
        if full_text.strip():
            agent_output = AgentOutput(
                session_id=session.session_id,
                intent=router_output.action,
                agent=agent.name,
                answer=full_text.strip(),
                language=router_output.language or Language.VI,
                sources=prepared.rag_result.sources if prepared.rag_result else [],
                metadata={"rag": prepared.rag_result.metadata if prepared.rag_result else {}},
            )
            try:
                await cache_service.save_from_agent_output(
                    router_output.action, cache_lookup_text, agent_output
                )
            except Exception as exc:
                logger.warning("Post-stream cache save failed: %s", exc)

            await session_store.add_assistant_message(session.session_id, full_text.strip())


chat_service = ChatService()
