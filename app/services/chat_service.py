import time

from app.agents.dispatcher import agent_dispatcher
from app.agents.router_agent import router_agent
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

logger = get_logger(__name__)


class ChatService:
    """
    End-to-end orchestration service.

    Main flow:

    request
    → session
    → router
    → dispatcher
    → agent
    → save history
    → response
    """

    async def chat(
        self,
        request: ChatRequest,
    ) -> ChatResponse:

        start = time.perf_counter()

        # -------
        # Session
        # -------
        session = await session_store.get_or_create(
            request.session_id
        )

        await session_store.add_user_message(
            session.session_id,
            request.text,
        )

        # ------
        # Router
        # ------
        router_output = await queue_manager.router.run(
            router_agent.classify,
            RouterInput(
                session_id=session.session_id,
                text=request.text,
                history=session.recent_history(),
                language=request.language,
            ),
        )

        # --------------
        # Dispatch agent
        # --------------
        agent = agent_dispatcher.get_agent(
            router_output.action
        )

        # ---------
        # Run agent
        # ---------
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
                    )
                },
            ),
        )

        # -----------------------
        # Save assistant response
        # -----------------------
        await session_store.add_assistant_message(
            session.session_id,
            agent_output.answer,
        )

        latency_ms = (
            time.perf_counter() - start
        ) * 1000

        logger.info(
            (
                "Chat done "
                "session=%s "
                "intent=%s "
                "latency=%.2fms"
            ),
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
            },
        )


chat_service = ChatService()