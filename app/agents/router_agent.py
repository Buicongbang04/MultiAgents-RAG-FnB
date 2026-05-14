from app.agents.intent_rules import classify_by_rules, detect_language
from app.core.constants import AgentName
from app.core.logging import get_logger
from app.core.schemas import RouterInput, RouterOutput

logger = get_logger(__name__)

class RouterAgent:
    """
    Router Agent MVP. """

    name = AgentName.ROUTER

    async def classify(self, router_input: RouterInput) -> RouterOutput:
        match = classify_by_rules(router_input.text)
        language = router_input.language or detect_language(router_input.text)

        output = RouterOutput(
            action=match.intent,
            confidence=match.score,
            language=language,
            raw_output=None,
            metadata={
                "router_type": "rule_based_mvp",
                "matched_keywords": match.matched_keywords,
                "reason": match.reason,
            },
        )

        logger.info(
            "Router classified session=%s intent=%s confidence=%.2f reason=%s",
            router_input.session_id,
            output.action.value,
            output.confidence,
            match.reason,
        )
        return output


router_agent = RouterAgent()