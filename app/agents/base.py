from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from app.core.schemas import AgentInput, AgentOutput, RAGResult
from app.llm.base import LLMGenerateRequest


@dataclass
class AgentPrepared:
    """Kết quả bước prepare: RAG + context đã xong, LLM chưa gọi."""
    llm_request: LLMGenerateRequest
    fallback_answer: str
    rag_result: Optional[RAGResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    name = None

    @abstractmethod
    async def prepare(self, agent_input: AgentInput) -> AgentPrepared:
        """RAG retrieval + context building. Không gọi LLM."""
        raise NotImplementedError

    async def run(self, agent_input: AgentInput) -> AgentOutput:
        """Blocking: prepare() → llm.generate() → AgentOutput."""
        from app.core.constants import Language
        from app.llm import get_llm_client

        prepared = await self.prepare(agent_input)
        llm = get_llm_client()
        llm_response = await llm.generate(prepared.llm_request)

        return AgentOutput(
            session_id=agent_input.session_id,
            intent=agent_input.intent,
            agent=self.name,
            answer=llm_response.text,
            language=agent_input.language or Language.VI,
            sources=prepared.rag_result.sources if prepared.rag_result else [],
            metadata={
                "rag": prepared.rag_result.metadata if prepared.rag_result else {},
                "llm": {
                    "backend": llm_response.backend,
                    "model": llm_response.model,
                    **llm_response.metadata,
                },
                **prepared.metadata,
            },
        )
