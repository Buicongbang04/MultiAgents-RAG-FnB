from app.agents.base import BaseAgent
from app.core.constants import AgentName, Intent, Language
from app.core.schemas import AgentInput, AgentOutput
from app.llm import LLMGenerateRequest, get_llm_client
from app.rag.retriever import graph_retriever

from app.prompts import FAQ_SYSTEM_PROMPT
from app.agents.context_builders import build_faq_structured_context

class FAQAgent(BaseAgent):
    name = AgentName.FAQ

    async def run(self, agent_input: AgentInput) -> AgentOutput:
        rag_result = await graph_retriever.retrieve(
            rag_query=agent_input.metadata["rag_query"]
        )

        structured_context = build_faq_structured_context(
            sources=rag_result.sources,
        )

        if not rag_result.has_context:
            fallback_answer = (
                "Dạ, hiện tại em chưa tìm thấy thông tin này trong dữ liệu FAQ/nội bộ. "
                "Anh/chị vui lòng hỏi lại nhân viên tại quầy để được xác nhận chính xác ạ."
            )
        else:
            fallback_answer = (
                "Dạ, theo thông tin hiện có trong hệ thống:\n"
                f"{rag_result.sources[0].text}\n\n"
                "Em chỉ trả lời dựa trên dữ liệu hiện có, không tự suy diễn thêm ạ."
            )

        llm = get_llm_client()
        llm_response = await llm.generate(
            LLMGenerateRequest(
                system_prompt=FAQ_SYSTEM_PROMPT,
                user_prompt=agent_input.text,
                context=structured_context,
                history=[
                    {"role": msg.role, "content": msg.content}
                    for msg in agent_input.history
                ],
                metadata={
                    "agent": self.name.value,
                    "intent": Intent.FAQ.value,
                    "fallback_answer": fallback_answer,
                    "rag": rag_result.metadata,
                },
            )
        )

        return AgentOutput(
            session_id=agent_input.session_id,
            intent=Intent.FAQ,
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


faq_agent = FAQAgent()