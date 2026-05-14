from app.agents.base import BaseAgent
from app.core.constants import AgentName, Intent, Language
from app.core.schemas import AgentInput, AgentOutput
from app.rag.retriever import graph_retriever


class FAQAgent(BaseAgent):
    name = AgentName.FAQ

    async def run(self, agent_input: AgentInput) -> AgentOutput:
        rag_result = await graph_retriever.retrieve(
            rag_query=agent_input.metadata["rag_query"]
        )

        if not rag_result.has_context:
            answer = (
                "Dạ, hiện tại em chưa tìm thấy thông tin này trong dữ liệu FAQ/nội bộ. "
                "Anh/chị vui lòng hỏi lại nhân viên tại quầy để được xác nhận chính xác ạ."
            )
        else:
            answer = (
                "Dạ, theo thông tin hiện có trong hệ thống:\n"
                f"{rag_result.sources[0].text}\n\n"
                "Em chỉ trả lời dựa trên dữ liệu hiện có, không tự suy diễn thêm ạ."
            )

        return AgentOutput(
            session_id=agent_input.session_id,
            intent=Intent.FAQ,
            agent=self.name,
            answer=answer,
            language=agent_input.language or Language.VI,
            sources=rag_result.sources,
            metadata={"rag": rag_result.metadata},
        )


faq_agent = FAQAgent()