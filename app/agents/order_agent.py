from app.agents.base import BaseAgent
from app.core.constants import AgentName, Intent, Language
from app.core.schemas import AgentInput, AgentOutput
from app.llm import LLMGenerateRequest, get_llm_client
from app.rag.retriever import graph_retriever

from app.prompts import ORDER_SYSTEM_PROMPT

# ======= Helper function =======

def format_vnd(price: int) -> str:
    return f"{price:,}".replace(",", ".") + "đ"

def build_order_structured_context(user_text: str, sources) -> str:
    if not sources:
        return ""

    user_text_lower = user_text.lower()

    exact_sources = [
        source for source in sources
        if source.text.split("|")[0].strip().lower() in user_text_lower
    ]

    best_source = exact_sources[0] if exact_sources else sources[0]

    price = best_source.metadata.get("price")
    size = best_source.metadata.get("size")
    category = best_source.metadata.get("category")

    best_lines = [
        "BEST_MATCH:",
        f"- id: {best_source.source_id}",
        f"- item: {best_source.text.split('|')[0].strip()}",
    ]

    if price:
        best_lines.append(f"- price: {format_vnd(price)}")

    if size:
        best_lines.append(f"- size: {size}")

    if category:
        best_lines.append(f"- category: {category}")

    alternative_lines = ["", "ALTERNATIVES:"]

    for source in sources:
        if source.source_id == best_source.source_id:
            continue

        item_name = source.text.split("|")[0].strip()
        alt_price = source.metadata.get("price")
        alt_size = source.metadata.get("size")

        line = f"- {item_name}"

        if alt_size:
            line += f" | size: {alt_size}"

        if alt_price:
            line += f" | price: {format_vnd(alt_price)}"

        alternative_lines.append(line)

    instruction_lines = [
        "",
        "ORDER_RULES:",
        "- Use BEST_MATCH as the primary item.",
        "- Do not confirm the order as completed.",
        "- Ask the customer to confirm before adding to cart.",
        "- Do not calculate total price.",
        "- Do not invent item, size, price, topping, or discount.",
    ]

    return "\n".join(best_lines + alternative_lines + instruction_lines)

# ===============================

class OrderAgent(BaseAgent):
    name = AgentName.ORDER

    async def run(self, agent_input: AgentInput) -> AgentOutput:
        rag_result = await graph_retriever.retrieve(
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
                context=structured_context,
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