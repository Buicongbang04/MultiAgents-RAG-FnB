from app.agents.base import AgentPrepared, BaseAgent
from app.core.constants import AgentName, Intent
from app.core.schemas import AgentInput
from app.llm.base import LLMGenerateRequest
from app.prompts import ORDER_SYSTEM_PROMPT
from app.rag.retriever import graph_retriever
from app.agents.context_builders import build_order_structured_context


class OrderAgent(BaseAgent):
    name = AgentName.ORDER

    async def prepare(self, agent_input: AgentInput) -> AgentPrepared:
        rag_result = await graph_retriever.retrieve_auto(
            rag_query=agent_input.metadata["rag_query"]
        )
        structured_context = build_order_structured_context(
            user_text=agent_input.text,
            sources=rag_result.sources,
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
            fallback_answer = f"Dạ, em tìm thấy món phù hợp trong menu:\n{top.text}\n"
            if price:
                fallback_answer += f"Giá: {price:,}đ"
            if size:
                fallback_answer += f" | Size: {size}"

        history = [{"role": m.role, "content": m.content} for m in agent_input.history]

        return AgentPrepared(
            llm_request=LLMGenerateRequest(
                system_prompt=ORDER_SYSTEM_PROMPT,
                user_prompt=agent_input.text,
                context=structured_context,
                history=history,
                metadata={
                    "agent": self.name.value,
                    "intent": Intent.ORDER.value,
                    "fallback_answer": fallback_answer,
                    "rag": rag_result.metadata,
                },
            ),
            fallback_answer=fallback_answer,
            rag_result=rag_result,
        )


order_agent = OrderAgent()
