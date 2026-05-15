from app.agents.base import BaseAgent
from app.core.constants import AgentName, Intent, Language, SourceType
from app.core.schemas import AgentInput, AgentOutput
from app.llm import LLMGenerateRequest, get_llm_client
from app.rag.retriever import graph_retriever

from app.prompts import CONSULTANT_SYSTEM_PROMPT
from app.agents.context_builders import build_consultant_structured_context


class ConsultantAgent(BaseAgent):
    name = AgentName.CONSULTANT

    async def run(self, agent_input: AgentInput) -> AgentOutput:
        rag_result = await graph_retriever.retrieve_auto(
            rag_query=agent_input.metadata["rag_query"]
        )

        structured_context = build_consultant_structured_context(
            sources=rag_result.sources,
        )

        if not rag_result.has_context:
            fallback_answer = (
                "Dạ, em chưa có đủ dữ liệu để tư vấn chính xác. "
                "Anh/chị có thể cho em biết khẩu vị mong muốn như ít ngọt, "
                "đậm cà phê, mát lạnh hoặc ngân sách khoảng bao nhiêu không ạ?"
            )
        else:
            menu_sources = [
                source
                for source in rag_result.sources
                if source.source_type == SourceType.MENU
            ]

            if menu_sources:
                lines = []
                for idx, src in enumerate(menu_sources[:3], start=1):
                    lines.append(f"{idx}. {src.text}")

                fallback_answer = (
                    "Dạ, dựa trên dữ liệu menu hiện có, em gợi ý một vài lựa chọn:\n"
                    + "\n".join(lines)
                    + "\n\nAnh/chị muốn em ưu tiên món ít ngọt, nhiều cà phê hay mát lạnh hơn ạ?"
                )
            else:
                fallback_answer = (
                    "Dạ, em tìm thấy một số thông tin tư vấn liên quan:\n"
                    f"{rag_result.context_text}\n\n"
                    "Anh/chị có thể nói rõ khẩu vị để em gợi ý món phù hợp hơn ạ."
                )

        llm = get_llm_client()
        llm_response = await llm.generate(
            LLMGenerateRequest(
                system_prompt=CONSULTANT_SYSTEM_PROMPT,
                user_prompt=agent_input.text,
                context=structured_context,
                history=[
                    {"role": msg.role, "content": msg.content}
                    for msg in agent_input.history
                ],
                metadata={
                    "agent": self.name.value,
                    "intent": Intent.CONSULTANT.value,
                    "fallback_answer": fallback_answer,
                    "rag": rag_result.metadata,
                },
            )
        )

        return AgentOutput(
            session_id=agent_input.session_id,
            intent=Intent.CONSULTANT,
            agent=self.name,
            answer=llm_response.text,
            language=agent_input.language or Language.VI,
            sources=rag_result.sources,
            metadata={
                "rag": rag_result.metadata,
                "llm": {
                    "backend": llm_response.backend,
                    "model": llm_response.model,
                    **llm_response.metadata,
                },
            },
        )


consultant_agent = ConsultantAgent()