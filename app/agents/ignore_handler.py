from app.agents.base import AgentPrepared, BaseAgent
from app.core.constants import AgentName, Intent
from app.core.schemas import AgentInput
from app.llm.base import LLMGenerateRequest
from app.prompts import IGNORE_SYSTEM_PROMPT


class IgnoreHandler(BaseAgent):
    name = AgentName.IGNORE

    async def prepare(self, agent_input: AgentInput) -> AgentPrepared:
        fallback_answer = (
            "Dạ em nghe ạ. Anh/chị muốn đặt món, hỏi thông tin quán, "
            "hay cần em gợi ý món phù hợp không ạ?"
        )
        history = [{"role": m.role, "content": m.content} for m in agent_input.history]

        return AgentPrepared(
            llm_request=LLMGenerateRequest(
                system_prompt=IGNORE_SYSTEM_PROMPT,
                user_prompt=agent_input.text,
                context=None,
                history=history,
                metadata={
                    "agent": self.name.value,
                    "intent": Intent.IGNORE.value,
                    "fallback_answer": fallback_answer,
                },
            ),
            fallback_answer=fallback_answer,
            rag_result=None,
        )


ignore_handler = IgnoreHandler()
