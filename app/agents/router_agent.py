from __future__ import annotations

from app.agents.intent_rules import classify_by_rules, detect_language
from app.core.config import get_settings
from app.core.constants import AgentName
from app.core.logging import get_logger
from app.core.schemas import RouterInput, RouterOutput
from app.agents.router_hf_lora import get_hf_lora_router, get_hf_merged_router

logger = get_logger(__name__)


class RouterAgent:
    """
    Router Agent.

    Supported backends:
    - rule_based: MVP baseline
    - hf_lora: Qwen2.5 router fine-tuned with LoRA

    Public output still satisfies:
    {"action": "<intent>"}
    """

    name = AgentName.ROUTER

    async def classify(self, router_input: RouterInput) -> RouterOutput:
        settings = get_settings()

        if settings.router_backend in {"hf_lora", "hf_merged"}:
            try:
                if settings.router_backend == "hf_merged":
                    router = get_hf_merged_router()
                else:
                    router = get_hf_lora_router()

                output = await router.classify(router_input)

                logger.info(
                    "Router classified session=%s backend=%s intent=%s",
                    router_input.session_id,
                    settings.router_backend,
                    output.action.value,
                )

                return output

            except Exception as exc:
                logger.exception(
                    "HF router failed. Falling back to rule_based. backend=%s error=%s",
                    settings.router_backend,
                    exc,
                )

                fallback = self._classify_rule_based(router_input)
                fallback.metadata["fallback_from"] = settings.router_backend
                fallback.metadata["fallback_error"] = str(exc)
                return fallback

        return self._classify_rule_based(router_input)

    def _classify_rule_based(self, router_input: RouterInput) -> RouterOutput:
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
                "required_json": {"action": match.intent.value},
            },
        )

        logger.info(
            "Router classified session=%s backend=rule_based intent=%s confidence=%.2f reason=%s",
            router_input.session_id,
            output.action.value,
            output.confidence,
            match.reason,
        )

        return output


router_agent = RouterAgent()