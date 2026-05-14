from app.agents.consultant_agent import consultant_agent
from app.agents.faq_agent import faq_agent
from app.agents.ignore_handler import ignore_handler
from app.agents.order_agent import order_agent
from app.core.constants import Intent


class AgentDispatcher:
    """
    Intent -> Agent mapping.
    """

    def __init__(self):
        self.agent_map = {
            Intent.ORDER: order_agent,
            Intent.CONSULTANT: consultant_agent,
            Intent.FAQ: faq_agent,
            Intent.IGNORE: ignore_handler,
        }

    def get_agent(self, intent: Intent):
        return self.agent_map[intent]


agent_dispatcher = AgentDispatcher()