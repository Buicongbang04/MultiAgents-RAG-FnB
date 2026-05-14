from app.agents.base import BaseAgent
from app.core.constants import AgentName, Intent, Language, SourceType
from app.core.schemas import AgentInput, AgentOutput
from app.rag.retriever import graph_retriever


class ConsultantAgent(BaseAgent):
    name = AgentName.CONSULTANT

    async def run(self, agent_input: AgentInput) -> AgentOutput:
        rag_result = await graph_retriever.retrieve(
            rag_query=agent_input.metadata["rag_query"]
        )

        if not rag_result.has_context:
            answer = (
                "Dạ, em chưa có đủ dữ liệu để tư vấn chính xác. "
                "Anh/chị có thể cho em biết khẩu vị mong muốn như ít ngọt, đậm cà phê, "
                "mát lạnh hoặc ngân sách khoảng bao nhiêu không ạ?"
            )
        else:
            menu_sources = [
                s for s in rag_result.sources
                if s.source_type == SourceType.MENU
            ]

            if menu_sources:
                lines = []
                for idx, src in enumerate(menu_sources[:3], start=1):
                    lines.append(f"{idx}. {src.text}")

                answer = (
                    "Dạ, dựa trên dữ liệu menu hiện có, em gợi ý một vài lựa chọn:\n"
                    + "\n".join(lines)
                    + "\n\nAnh/chị muốn em ưu tiên món ít ngọt, nhiều cà phê hay mát lạnh hơn ạ?"
                )
            else:
                answer = (
                    "Dạ, em tìm thấy một số thông tin tư vấn liên quan:\n"
                    f"{rag_result.context_text}\n\n"
                    "Anh/chị có thể nói rõ khẩu vị để em gợi ý món phù hợp hơn ạ."
                )

        return AgentOutput(
            session_id=agent_input.session_id,
            intent=Intent.CONSULTANT,
            agent=self.name,
            answer=answer,
            language=agent_input.language or Language.VI,
            sources=rag_result.sources,
            metadata={"rag": rag_result.metadata},
        )


consultant_agent = ConsultantAgent()