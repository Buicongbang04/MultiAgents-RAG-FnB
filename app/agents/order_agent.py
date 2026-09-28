from app.agents.base import AgentPrepared, BaseAgent
from app.agents.context_builders import (
    build_order_structured_context,
    format_price_range,
    format_sizes,
    format_vnd,
)
from app.agents.order_resolution import OrderAction, OrderResolution, resolve_order
from app.core.constants import AgentName, Intent
from app.core.schemas import AgentInput
from app.llm.base import LLMGenerateRequest
from app.prompts import ORDER_SYSTEM_PROMPT
from app.rag.retriever import graph_retriever


def build_order_fallback(resolution: OrderResolution) -> str:
    """Câu trả lời tất định theo NEXT_ACTION (dùng khi mock / LLM lỗi)."""
    action = resolution.action

    if action == OrderAction.NO_MATCH and resolution.unknown_words:
        requested = resolution.requested_phrase or " ".join(resolution.unknown_words)
        return (
            f"Dạ, bên em hiện chưa có món \"{requested}\" ạ. "
            "Anh/chị tham khảo cà phê, trà, freeze hoặc bánh bên em nhé, "
            "em gợi ý món hợp khẩu vị cho mình được không ạ?"
        )

    if action == OrderAction.NO_MATCH:
        return (
            "Dạ, em chưa tìm thấy món này trong menu hiện tại. "
            "Anh/chị có thể nói rõ tên món hơn được không ạ?"
        )

    # "có bán X không?" → trả lời "Dạ có ạ!" trước, như nhân viên thật.
    yes = "Dạ có ạ! " if resolution.availability_question else "Dạ, "

    if action == OrderAction.ASK_ITEM:
        options = "\n".join(
            f"• **{name}** — {format_price_range(resolution.candidate_variants.get(name) or [])}"
            for name in resolution.candidates
        )
        intro = "Bên em có các món:" if resolution.availability_question else "bên em có mấy món gần giống:"
        return f"{yes}{intro}\n{options}\n\nAnh/chị muốn gọi món nào ạ?"

    name = resolution.item_name
    quantity = resolution.quantity

    if action == OrderAction.ASK_SIZE:
        sizes = "\n".join(f"• Size **{v['size']}** — {format_vnd(v['price'])}" for v in resolution.variants)
        return f"{yes}**{name}** bên em có các size:\n{sizes}\n\nAnh/chị dùng size nào ạ?"

    if action == OrderAction.SIZE_UNAVAILABLE:
        return (
            f"Dạ, **{name}** không có size {resolution.size} ạ. "
            f"Bên em có: {format_sizes(resolution.variants)}. Anh/chị chọn size nào ạ?"
        )

    selected = resolution.selected
    size_part = f" size {selected['size']}" if len(resolution.variants) > 1 else ""
    if resolution.availability_question:
        return (
            f"Dạ có ạ! **{name}{size_part}** — {format_vnd(selected['price'])}. "
            f"Em lên {quantity} {'ly' if len(resolution.variants) > 1 else 'phần'} "
            "cho anh/chị luôn nhé? 😊"
        )
    return (
        f"Dạ, em ghi nhận **{quantity} {name}{size_part}** — {format_vnd(selected['price'])}"
        f"{'/ly' if len(resolution.variants) > 1 else ''}. "
        "Anh/chị xác nhận giúp em nhé? 😊"
    )


class OrderAgent(BaseAgent):
    name = AgentName.ORDER

    async def prepare(self, agent_input: AgentInput) -> AgentPrepared:
        rag_query = agent_input.metadata["rag_query"]
        rag_result = await graph_retriever.retrieve_auto(rag_query=rag_query)

        # Standalone query: follow-up "size M" đã được ghép với món của lượt trước.
        resolution = resolve_order(rag_query.query, rag_result.sources, latest_text=agent_input.text)
        structured_context = build_order_structured_context(resolution)
        fallback_answer = build_order_fallback(resolution)

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
            metadata={"order": resolution.to_metadata()},
        )


order_agent = OrderAgent()
