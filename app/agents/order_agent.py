from app.agents.base import BaseAgent
from app.core.constants import AgentName, Intent, Language
from app.core.schemas import AgentInput, AgentOutput
from app.rag.retriever import graph_retriever


class OrderAgent(BaseAgent):
    name = AgentName.ORDER

    async def run(self, agent_input: AgentInput) -> AgentOutput:
        rag_result = await graph_retriever.retrieve(
            rag_query=agent_input.metadata["rag_query"]
        )

        if not rag_result.has_context:
            answer = (
                "Dạ, em chưa tìm thấy món này trong menu hiện tại. "
                "Anh/chị có thể nói rõ tên món hơn được không ạ?"
            )
        else:
            top = rag_result.sources[0]
            price = top.metadata.get("price")
            size = top.metadata.get("size")
            category = top.metadata.get("category")

            answer = (
                "Dạ, em tìm thấy món phù hợp trong menu:\n"
                f"{top.text}\n"
            )

            if price:
                answer += f"Giá hiện tại: {price:,}đ"
                if size:
                    answer += f" | Size: {size}"
                if category:
                    answer += f" | Nhóm: {category}"
                answer += "\n"

            answer += (
                "MVP hiện tại chưa bật giỏ hàng, nên em chỉ xác nhận thông tin món trước ạ."
            )

        return AgentOutput(
            session_id=agent_input.session_id,
            intent=Intent.ORDER,
            agent=self.name,
            answer=answer,
            language=agent_input.language or Language.VI,
            sources=rag_result.sources,
            metadata={"rag": rag_result.metadata},
        )


order_agent = OrderAgent()