from app.agents.base import AgentPrepared, BaseAgent
from app.core.constants import AgentName, Intent, SourceType
from app.core.schemas import AgentInput
from app.llm.base import LLMGenerateRequest
from app.prompts import CONSULTANT_SYSTEM_PROMPT
from app.rag.retriever import graph_retriever
from app.agents.context_builders import build_consultant_structured_context


class ConsultantAgent(BaseAgent):
    name = AgentName.CONSULTANT

    async def prepare(self, agent_input: AgentInput) -> AgentPrepared:
        rag_result = await graph_retriever.retrieve_auto(
            rag_query=agent_input.metadata["rag_query"]
        )
        structured_context = build_consultant_structured_context(
            sources=rag_result.sources,
        )

        if not rag_result.has_context:
            fallback_answer = (
                "Dạ, em chưa có đủ dữ liệu để tư vấn chính xác. "
                "Anh/chị cho em biết khẩu vị, ngân sách hoặc thời tiết hôm nay nhé ạ?"
            )
        else:
            menu_sources = [s for s in rag_result.sources if s.source_type == SourceType.MENU]
            if menu_sources:
                lines = [f"{i}. {s.text}" for i, s in enumerate(menu_sources[:3], 1)]
                fallback_answer = (
                    "Dạ, dựa trên menu hiện có, em gợi ý:\n"
                    + "\n".join(lines)
                    + "\n\nAnh/chị muốn ít ngọt, nhiều cà phê hay mát lạnh hơn ạ?"
                )
            else:
                fallback_answer = (
                    f"Dạ, em tìm thấy:\n{rag_result.context_text}\n\n"
                    "Anh/chị nói rõ khẩu vị để em gợi ý phù hợp hơn ạ."
                )

        history = [{"role": m.role, "content": m.content} for m in agent_input.history]

        return AgentPrepared(
            llm_request=LLMGenerateRequest(
                system_prompt=CONSULTANT_SYSTEM_PROMPT,
                user_prompt=agent_input.text,
                context=structured_context,
                history=history,
                metadata={
                    "agent": self.name.value,
                    "intent": Intent.CONSULTANT.value,
                    "fallback_answer": fallback_answer,
                    "rag": rag_result.metadata,
                },
            ),
            fallback_answer=fallback_answer,
            rag_result=rag_result,
        )


consultant_agent = ConsultantAgent()
