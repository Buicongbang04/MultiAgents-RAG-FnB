from app.agents.base import AgentPrepared, BaseAgent
from app.core.constants import AgentName, Intent
from app.core.schemas import AgentInput
from app.llm.base import LLMGenerateRequest
from app.prompts import FAQ_SYSTEM_PROMPT
from app.rag.retriever import graph_retriever
from app.agents.context_builders import build_faq_structured_context


class FAQAgent(BaseAgent):
    name = AgentName.FAQ

    async def prepare(self, agent_input: AgentInput) -> AgentPrepared:
        rag_result = await graph_retriever.retrieve_auto(
            rag_query=agent_input.metadata["rag_query"]
        )
        structured_context = build_faq_structured_context(sources=rag_result.sources)

        if not rag_result.has_context:
            fallback_answer = (
                "Dạ, hiện tại em chưa tìm thấy thông tin này trong FAQ. "
                "Anh/chị vui lòng hỏi nhân viên tại quầy để được xác nhận chính xác ạ."
            )
        else:
            fallback_answer = (
                f"Dạ, theo dữ liệu hiện có:\n{rag_result.sources[0].text}\n\n"
                "Em chỉ trả lời dựa trên dữ liệu có sẵn, không tự suy diễn thêm ạ."
            )

        history = [{"role": m.role, "content": m.content} for m in agent_input.history]

        return AgentPrepared(
            llm_request=LLMGenerateRequest(
                system_prompt=FAQ_SYSTEM_PROMPT,
                user_prompt=agent_input.text,
                context=structured_context,
                history=history,
                metadata={
                    "agent": self.name.value,
                    "intent": Intent.FAQ.value,
                    "fallback_answer": fallback_answer,
                    "rag": rag_result.metadata,
                },
            ),
            fallback_answer=fallback_answer,
            rag_result=rag_result,
        )


faq_agent = FAQAgent()
