from app.agents.base import BaseAgent
from app.core.constants import AgentName, Intent, Language
from app.core.schemas import AgentInput, AgentOutput


class IgnoreHandler(BaseAgent):
    name = AgentName.IGNORE

    async def run(self, agent_input: AgentInput) -> AgentOutput:
        answer = (
            "Dạ em nghe ạ. Anh/chị muốn đặt món, hỏi thông tin quán, "
            "hay cần em gợi ý món phù hợp không ạ?"
        )

        return AgentOutput(
            session_id=agent_input.session_id,
            intent=Intent.IGNORE,
            agent=self.name,
            answer=answer,
            language=agent_input.language or Language.VI,
            sources=[],
            metadata={"handler": "rule_based_ignore"},
        )


ignore_handler = IgnoreHandler()