from app.agents.base import AgentPrepared, BaseAgent
from app.core.config import get_settings
from app.core.constants import AgentName, Intent
from app.core.schemas import AgentInput
from app.llm.base import LLMGenerateRequest
from app.prompts import CONSULTANT_SYSTEM_PROMPT
from app.rag.retriever import graph_retriever
from app.agents.context_builders import (
    build_consultant_structured_context,
    format_sizes,
    short_description,
)
from app.agents.order_resolution import distinct_menu_items
from app.rag.menu import get_menu_variants


class ConsultantAgent(BaseAgent):
    name = AgentName.CONSULTANT

    async def prepare(self, agent_input: AgentInput) -> AgentPrepared:
        rag_result = await graph_retriever.retrieve_auto(
            rag_query=agent_input.metadata["rag_query"]
        )
        # Mỗi size là một node → gộp theo tên món, lấy đủ size/giá từ graph.
        context_config = get_settings().rag.context
        item_names = distinct_menu_items(rag_result.sources)[: context_config.consultant_max_menu_items]
        variants = get_menu_variants(item_names)

        structured_context = build_consultant_structured_context(
            sources=rag_result.sources,
            variants=variants,
        )

        if not rag_result.has_context:
            fallback_answer = (
                "Dạ, em chưa có đủ dữ liệu để tư vấn chính xác. "
                "Anh/chị cho em biết khẩu vị, ngân sách hoặc thời tiết hôm nay nhé ạ?"
            )
        elif item_names:
            lines = []
            for name in item_names[: context_config.consultant_fallback_items]:
                item_variants = variants.get(name) or []
                if not item_variants:
                    continue
                note = short_description(item_variants[0]["description"])
                line = f"• **{name}**"
                if note:
                    line += f" — {note}"
                line += f"\n   {format_sizes(item_variants)}"
                lines.append(line)
            fallback_answer = (
                "Dạ, em gợi ý mình vài món nè:\n"
                + "\n".join(lines)
                + "\n\nAnh/chị ưng món nào để em lên đơn ạ? Hoặc cho em biết thêm khẩu vị "
                "(đậm vị, mát lạnh, giá mềm…) để em gợi ý sát hơn nhé."
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
