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
from app.services.conversation import conversation_resolver
from app.services.order_state import order_dialogue_manager
from app.session.session_store import session_store
from app.streaming.sse import sse_event
from app.streaming.clause_splitter import ClauseSplitter

logger = get_logger(__name__)


class ChatService:
    """
    End-to-end orchestration.

    Flow (non-streaming):
      request → session → order dialogue state ("ok", "xem đơn"… → trả lời ngay)
      → router → follow-up resolver → intent extractor → exact cache
      → semantic cache → agent.run() → save cache → save history → response

    Flow (streaming):
      request → session → router → intent extractor → exact/semantic cache
      → if hit: stream cached answer
      → if miss: agent.prepare() → llm.stream_generate() → pipe SSE
      → save cache + history after stream completes
    """

    async def _resolve_cache_lookup(
        self,
        query: str,
        session_id: str,
        router_output,
        history,
    ):
        """Run intent extractor + cache lookup. Returns (extraction, cache_lookup_text, cached)."""
        settings = get_settings()
        extraction = None
        cache_lookup_text = query

        if settings.intent_extractor.enabled:
            try:
                extraction = await get_intent_extractor().extract(
                    IntentExtractionInput(
                        session_id=session_id,
                        text=query,
                        intent=router_output.action,
                        language=router_output.language,
                        history=history,
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

    async def _handle_dialogue_turn(self, session, text: str):
        """Order dialogue state chạy trước router. Trả về DialogueTurn hoặc None."""
        turn = order_dialogue_manager.handle(session, text)
        if turn is None:
            return None
        await session_store.add_assistant_message(session.session_id, turn.answer)
        conversation_resolver.mark_dialogue_turn(session, turn.topic)
        logger.info("Order dialogue session=%s action=%s", session.session_id, turn.action)
        return turn

    @staticmethod
    def _dialogue_metadata(turn) -> dict:
        return {
            "router": {"router_type": "order_dialogue_state", "reason": turn.action},
            "order_dialogue": turn.metadata,
        }

    async def chat(self, request: ChatRequest) -> ChatResponse:
        start = time.perf_counter()

        session = await session_store.get_or_create(request.session_id)
        # Snapshot trước khi thêm câu hiện tại: câu hỏi đã nằm trong user_prompt,
        # nếu để trong history nữa thì LLM nhận nó 2 lần.
        history = session.recent_history()
        await session_store.add_user_message(session.session_id, request.text)

        dialogue_turn = await self._handle_dialogue_turn(session, request.text)
        if dialogue_turn is not None:
            return ChatResponse(
                session_id=session.session_id,
                intent=Intent.ORDER,
                agent=AgentName.ORDER,
                answer=dialogue_turn.answer,
                language=Language.VI,
                latency_ms=(time.perf_counter() - start) * 1000,
                metadata=self._dialogue_metadata(dialogue_turn),
            )

        router_output = await queue_manager.router.run(
            router_agent.classify,
            RouterInput(
                session_id=session.session_id,
                text=request.text,
                history=history,
                language=request.language,
            ),
        )

        resolved = await conversation_resolver.resolve(session, request.text, router_output, history)
        router_output = resolved.apply_to(router_output)

        extraction, cache_lookup_text, cached = await self._resolve_cache_lookup(
            resolved.query, session.session_id, router_output, history
        )

        if cached is not None:
            if cached["metadata"].get("cache_type") == "exact":
                try:
                    await cache_service.backfill_semantic_from_cached_value(
                        router_output.action, cache_lookup_text, cached["value"]
                    )
                except Exception as exc:
                    logger.warning("Semantic backfill failed: %s", exc)

            reminder = order_dialogue_manager.pending_reminder(session, router_output.action)
            await session_store.add_assistant_message(
                session.session_id, cached["value"]["answer"]
            )
            conversation_resolver.remember(session, resolved)
            order_dialogue_manager.after_agent(session, router_output.action, None)
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
            response.metadata["conversation"] = resolved.metadata
            if reminder:
                response.answer = f"{response.answer}\n\n{reminder}"
            return response

        agent = agent_dispatcher.get_agent(router_output.action)
        agent_output = await queue_manager.generator.run(
            agent.run,
            AgentInput(
                session_id=session.session_id,
                text=request.text,
                intent=router_output.action,
                history=history,
                language=router_output.language,
                metadata={
                    "rag_query": RAGQuery(
                        query=resolved.query,
                        intent=router_output.action,
                        language=router_output.language,
                    ),
                    "session_summary": session.summary,
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

        reminder = order_dialogue_manager.pending_reminder(session, router_output.action)
        if reminder:
            agent_output.answer = f"{agent_output.answer}\n\n{reminder}"

        await session_store.add_assistant_message(session.session_id, agent_output.answer)
        order_metadata = agent_output.metadata.get("order")
        conversation_resolver.remember(session, resolved, (order_metadata or {}).get("action"))
        order_dialogue_manager.after_agent(session, router_output.action, order_metadata)
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
                "conversation": resolved.metadata,
            },
        )

    @staticmethod
    async def _stream_text(answer: str) -> AsyncGenerator[str, None]:
        """Stream một câu trả lời có sẵn (cache hit / order dialogue) theo cùng giao thức SSE."""
        splitter = ClauseSplitter()
        for word in answer.split():
            token = word + " "
            yield sse_event(StreamEventType.TOKEN, {"token": token})
            for clause in splitter.push(token):
                yield sse_event(StreamEventType.CLAUSE, {"clause": clause})

        for clause in splitter.flush():
            yield sse_event(StreamEventType.CLAUSE, {"clause": clause})

        yield "data: [DONE]\n\n"

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
        # Snapshot trước khi thêm câu hiện tại: câu hỏi đã nằm trong user_prompt,
        # nếu để trong history nữa thì LLM nhận nó 2 lần.
        history = session.recent_history()
        await session_store.add_user_message(session.session_id, request.text)

        dialogue_turn = await self._handle_dialogue_turn(session, request.text)
        if dialogue_turn is not None:
            yield sse_event(
                StreamEventType.METADATA,
                {
                    "ttft_ms": round((time.perf_counter() - start_time) * 1000, 2),
                    "session_id": session.session_id,
                    "intent": Intent.ORDER.value,
                    "cache_hit": False,
                    **self._dialogue_metadata(dialogue_turn),
                },
            )
            async for event in self._stream_text(dialogue_turn.answer):
                yield event
            return

        try:
            router_output = await queue_manager.router.run(
                router_agent.classify,
                RouterInput(
                    session_id=session.session_id,
                    text=request.text,
                    history=history,
                    language=request.language,
                ),
            )
        except Exception as exc:
            yield sse_event(StreamEventType.ERROR, {"message": str(exc)})
            yield "data: [DONE]\n\n"
            return

        resolved = await conversation_resolver.resolve(session, request.text, router_output, history)
        router_output = resolved.apply_to(router_output)

        extraction, cache_lookup_text, cached = await self._resolve_cache_lookup(
            resolved.query, session.session_id, router_output, history
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
                    "conversation": resolved.metadata,
                },
            )

            reminder = order_dialogue_manager.pending_reminder(session, router_output.action)
            if reminder:
                answer = f"{answer}\n\n{reminder}"
            async for event in self._stream_text(answer):
                yield event

            await session_store.add_assistant_message(session.session_id, answer)
            order_dialogue_manager.after_agent(session, router_output.action, None)
            conversation_resolver.remember(session, resolved)
            return

        # ── Cache miss: prepare agent + stream LLM ───────────────────────────
        agent = agent_dispatcher.get_agent(router_output.action)

        try:
            agent_input = AgentInput(
                session_id=session.session_id,
                text=request.text,
                intent=router_output.action,
                history=history,
                language=router_output.language,
                metadata={
                    "rag_query": RAGQuery(
                        query=resolved.query,
                        intent=router_output.action,
                        language=router_output.language,
                    ),
                    "session_summary": session.summary,
                },
            )
            prepared = await agent.prepare(agent_input)
            prepared.llm_request.summary = session.summary
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
                            "conversation": resolved.metadata,
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

        cacheable_text = full_text
        reminder = order_dialogue_manager.pending_reminder(session, router_output.action)
        if reminder and full_text.strip():
            for word in f"\n\n{reminder}".split(" "):
                token = word + " "
                full_text += token
                yield sse_event(StreamEventType.TOKEN, {"token": token})
                for clause in splitter.push(token):
                    yield sse_event(StreamEventType.CLAUSE, {"clause": clause})

        for clause in splitter.flush():
            yield sse_event(StreamEventType.CLAUSE, {"clause": clause})

        yield "data: [DONE]\n\n"

        # ── Post-stream: save cache + history ────────────────────────────────
        if full_text.strip():
            agent_output = AgentOutput(
                session_id=session.session_id,
                intent=router_output.action,
                agent=agent.name,
                answer=cacheable_text.strip(),
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
            order_metadata = prepared.metadata.get("order")
            conversation_resolver.remember(session, resolved, (order_metadata or {}).get("action"))
            order_dialogue_manager.after_agent(session, router_output.action, order_metadata)


chat_service = ChatService()
