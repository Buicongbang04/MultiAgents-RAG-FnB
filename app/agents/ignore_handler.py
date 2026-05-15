from app.agents.base import BaseAgent
from app.core.constants import AgentName, Intent, Language
from app.core.schemas import AgentInput, AgentOutput
from app.llm import LLMGenerateRequest, get_llm_client

from app.prompts import IGNORE_SYSTEM_PROMPT

class IgnoreHandler(BaseAgent):
    name = AgentName.IGNORE

    async def run(self, agent_input: AgentInput) -> AgentOutput:
        fallback_answer = (
            "Dạ em nghe ạ. Anh/chị muốn đặt món, hỏi thông tin quán, "
            "hay cần em gợi ý món phù hợp không ạ?"
        )

        llm = get_llm_client()
        llm_response = await llm.generate(
            LLMGenerateRequest(
                system_prompt=IGNORE_SYSTEM_PROMPT,
                user_prompt=agent_input.text,
                context=None,
                history=[
                    {"role": msg.role, "content": msg.content}
                    for msg in agent_input.history
                ],
                metadata={
                    "agent": self.name.value,
                    "intent": Intent.IGNORE.value,
                    "fallback_answer": fallback_answer,
                    "handler": "llm_client_ignore",
                },
            )
        )

        return AgentOutput(
            session_id=agent_input.session_id,
            intent=Intent.IGNORE,
            agent=self.name,
            answer=llm_response.text,
            language=agent_input.language or Language.VI,
            sources=[],
            metadata={
                "handler": "llm_client_ignore",
                "llm": {
                    "backend": llm_response.backend,
                    "model": llm_response.model,
                    **llm_response.metadata,
                },
            },
        )


ignore_handler = IgnoreHandler()