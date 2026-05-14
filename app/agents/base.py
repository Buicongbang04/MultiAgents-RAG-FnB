from abc import ABC, abstractmethod

from app.core.schemas import AgentInput, AgentOutput


class BaseAgent(ABC):
    name = None

    @abstractmethod
    async def run(self, agent_input: AgentInput) -> AgentOutput:
        raise NotImplementedError