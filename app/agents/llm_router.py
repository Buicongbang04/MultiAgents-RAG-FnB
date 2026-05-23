"""
LLM-based intent router.

Gọi SGLang/vLLM với prompt rất ngắn để phân loại ý định.
max_tokens=10 → chỉ trả về label, latency thấp (<200ms).
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict

import httpx

from app.core.constants import Intent, Language
from app.core.logging import get_logger
from app.core.schemas import RouterInput, RouterOutput

logger = get_logger(__name__)

INTENT_LABELS = {"order", "consultant", "faq", "ignore"}

SYSTEM_PROMPT = """\
Bạn là bộ phân loại ý định cho chatbot quán cà phê Highlands Coffee.

Phân loại tin nhắn khách vào 1 trong 4 nhãn:
- order: khách muốn đặt/gọi/order/mua đồ uống hoặc đồ ăn (kể cả chỉ nói tên món)
- consultant: khách muốn tư vấn, gợi ý, hỏi nên uống gì, có gì ngon
- faq: khách hỏi thông tin quán (wifi, giờ mở cửa, thanh toán, chính sách, giao hàng...)
- ignore: chào hỏi đơn thuần, tin nhắn nhiễu, không liên quan đến quán

Chỉ trả về đúng 1 từ: order | consultant | faq | ignore"""


def _parse_intent(raw: str) -> Intent | None:
    text = raw.strip().lower()
    for label in INTENT_LABELS:
        if label in text:
            return Intent(label)
    return None


class LLMIntentRouter:
    """
    Phân loại intent bằng LLM (SGLang/vLLM).
    Fallback về rule_based nếu LLM không phản hồi.
    """

    def __init__(self) -> None:
        from app.core.config import get_settings
        self.settings = get_settings()
        self.base_url = self.settings.llm_base_url.rstrip("/")
        self.model = self.settings.generator_model
        self.api_key = self.settings.llm_api_key

    async def classify(self, router_input: RouterInput) -> RouterOutput:
        started = time.perf_counter()
        intent = await self._call_llm(router_input.text)
        latency = round((time.perf_counter() - started) * 1000, 1)

        if intent is None:
            return self._fallback(router_input, latency)

        language = _detect_language(router_input.text)

        logger.info(
            "LLMRouter session=%s intent=%s latency=%.0fms",
            router_input.session_id,
            intent.value,
            latency,
        )

        return RouterOutput(
            action=intent,
            confidence=0.90,
            language=language,
            raw_output=None,
            metadata={
                "router_type": "llm",
                "matched_keywords": [],
                "reason": f"llm classification ({latency:.0f}ms)",
                "required_json": {"action": intent.value},
            },
        )

    async def _call_llm(self, text: str) -> Intent | None:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            "temperature": 0.0,
            "max_tokens": 10,
            "stream": False,
        }

        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                )
                resp.raise_for_status()
                raw = (
                    resp.json()
                    .get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                return _parse_intent(raw)
        except Exception as exc:
            logger.debug("LLMRouter LLM call failed: %s", exc)
            return None

    def _fallback(self, router_input: RouterInput, latency: float) -> RouterOutput:
        from app.agents.intent_rules import classify_by_rules, detect_language as detect_lang
        match = classify_by_rules(router_input.text)
        language = detect_lang(router_input.text)

        logger.info(
            "LLMRouter fallback to rule_based session=%s intent=%s",
            router_input.session_id,
            match.intent.value,
        )

        return RouterOutput(
            action=match.intent,
            confidence=match.score,
            language=language,
            raw_output=None,
            metadata={
                "router_type": "llm_fallback_rule_based",
                "matched_keywords": match.matched_keywords,
                "reason": f"llm unavailable ({latency:.0f}ms), rule_based fallback",
                "required_json": {"action": match.intent.value},
            },
        )


def _detect_language(text: str) -> Language:
    vi_chars = "àáâãèéêìíòóôõùúýăđơưạảấầẩẫậắặẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỷỹỵ"
    if any(c in text.lower() for c in vi_chars):
        return Language.VI
    if re.search(r"\b(hello|hi|order|coffee|tea|recommend|what|which|where)\b", text.lower()):
        return Language.EN
    return Language.UNKNOWN


_instance: LLMIntentRouter | None = None


def get_llm_router() -> LLMIntentRouter:
    global _instance
    if _instance is None:
        _instance = LLMIntentRouter()
    return _instance
