from app.agents.base import BaseAgent
from app.core.constants import AgentName, Intent, Language
from app.core.schemas import AgentInput, AgentOutput
from app.llm import LLMGenerateRequest, get_llm_client
from app.rag.retriever import graph_retriever

from app.prompts import ORDER_SYSTEM_PROMPT


class OrderAgent(BaseAgent):
    name = AgentName.ORDER

    async def run(self, agent_input: AgentInput) -> AgentOutput:
        rag_result = await graph_retriever.retrieve(
            rag_query=agent_input.metadata["rag_query"]
        )

        if not rag_result.has_context:
            fallback_answer = (
                "Dạ, em chưa tìm thấy món này trong menu hiện tại. "
                "Anh/chị có thể nói rõ tên món hơn được không ạ?"
            )
        else:
            top = rag_result.sources[0]
            price = top.metadata.get("price")
            size = top.metadata.get("size")
            category = top.metadata.get("category")

            fallback_answer = (
                "Dạ, em tìm thấy món phù hợp trong menu:\n"
                f"{top.text}\n"
            )

            if price:
                fallback_answer += f"Giá hiện tại: {price:,}đ"

            if size:
                fallback_answer += f" | Size: {size}"

            if category:
                fallback_answer += f" | Nhóm: {category}"

            fallback_answer += "\n"
            fallback_answer += (
                "MVP hiện tại chưa bật giỏ hàng, nên em chỉ xác nhận thông tin món trước ạ."
            )

        llm = get_llm_client()
        llm_response = await llm.generate(
            LLMGenerateRequest(
                system_prompt=ORDER_SYSTEM_PROMPT,
                user_prompt=agent_input.text,
                context=rag_result.context_text,
                history=[
                    {"role": msg.role, "content": msg.content}
                    for msg in agent_input.history
                ],
                metadata={
                    "agent": self.name.value,
                    "intent": Intent.ORDER.value,
                    "fallback_answer": fallback_answer,
                    "rag": rag_result.metadata,
                },
            )
        )

        return AgentOutput(
            session_id=agent_input.session_id,
            intent=Intent.ORDER,
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


order_agent = OrderAgent()