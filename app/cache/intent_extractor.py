from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from app.cache.exact_cache import normalize_cache_text
from app.core.config import get_settings
from app.core.constants import Intent, Language
from app.core.schemas import Message


class IntentExtractionInput(BaseModel):
    session_id: str
    text: str
    intent: Intent
    language: Language = Language.UNKNOWN
    history: List[Message] = Field(default_factory=list)


class IntentExtractionOutput(BaseModel):
    subject: str = ""
    action: str = ""
    context: str = ""
    cache_key: str = ""
    intent: Intent
    language: Language = Language.UNKNOWN
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseIntentExtractor(ABC):
    @abstractmethod
    async def extract(self, extractor_input: IntentExtractionInput) -> IntentExtractionOutput:
        raise NotImplementedError


class RuleBasedIntentExtractor(BaseIntentExtractor):
    """
    Phase 2.7.1 baseline extractor.

    Mục tiêu:
    - Tách action/context/cache_key đủ tốt để validate cache architecture.
    - Không thay router.
    - Không cache order.
    - Không dùng LLM, latency rất thấp.
    """

    SUBJECT_PATTERNS: Tuple[str, ...] = (
        r"\bcho\s+(anh|chị|em|mình|tôi|tui|bọn em|tụi em)\b",
        r"\b(anh|chị|em|mình|tôi|tui)\s+(muốn|cần|hỏi|xin|đặt|gọi|lấy|order)\b",
        r"\b(can i|could i|may i|i want|i need|i would like)\b",
    )

    CONTEXT_PATTERNS: Tuple[str, ...] = (
        r"\b(hôm nay|ngày mai|mai|tối nay|sáng nay|chiều nay|trưa nay)\b",
        r"\b(\d{1,2}h|\d{1,2}\s*giờ|\d{1,2}:\d{2})\b",
        r"\b(sinh nhật|họp|đi làm|đi học|mang đi|take away|takeaway|delivery|ship)\b",
        r"\b(\d+\s*(người|ly|cốc|phần|bạn))\b",
        r"\b(ít ngọt|không đường|nhiều đá|ít đá|nóng|đá|size\s*[sml]|size\s*[SML])\b",
        r"\b(giá mềm|giá rẻ|rẻ|budget|affordable|cheap|dễ uống)\b",
    )

    FILLER_PATTERNS: Tuple[str, ...] = (
        r"^(dạ|ạ|nha|nhé|với|giúp|giúp em|cho em|cho anh|cho chị)\s+",
        r"\s+(nha|nhé|ạ|với|giúp em|giúp anh|được không|được hông)\s*$",
    )

    def _detect_language(self, text: str, fallback: Language) -> Language:
        if fallback and fallback != Language.UNKNOWN:
            return fallback

        normalized = text.lower()
        vi_markers = [
            "đ", "ă", "â", "ê", "ô", "ơ", "ư",
            "anh", "chị", "em", "mình", "quán", "món", "giá", "ngon",
            "mật khẩu", "gợi ý", "đóng cửa", "mở cửa",
        ]
        if any(marker in normalized for marker in vi_markers):
            return Language.VI
        return Language.EN

    def _extract_subject(self, text: str, language: Language) -> str:
        normalized = text.lower().strip()

        for pattern in self.SUBJECT_PATTERNS:
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if match:
                if match.lastindex:
                    return match.group(1).strip()
                return "I" if language == Language.EN else ""

        if language == Language.EN and re.search(r"\b(i|me|my)\b", normalized):
            return "I"

        return ""

    def _extract_context(self, text: str) -> str:
        normalized = text.lower()
        contexts: List[str] = []

        for pattern in self.CONTEXT_PATTERNS:
            for match in re.finditer(pattern, normalized, flags=re.IGNORECASE):
                value = match.group(0).strip()
                if value and value not in contexts:
                    contexts.append(value)

        return ", ".join(contexts)

    def _clean_action_text(self, text: str) -> str:
        action = normalize_cache_text(text)

        for pattern in self.FILLER_PATTERNS:
            action = re.sub(pattern, "", action, flags=re.IGNORECASE).strip()

        action = re.sub(
            r"\b(dạ|ạ|nha|nhé|với|giúp|em hỏi xíu|hỏi xíu|cho hỏi|xin hỏi)\b",
            " ",
            action,
            flags=re.IGNORECASE,
        )
        action = re.sub(r"\s+", " ", action).strip()
        return action

    def _classify_action_cache_key(
        self,
        text: str,
        intent: Intent,
        language: Language,
    ) -> Tuple[str, str, float, Dict[str, Any]]:
        normalized = normalize_cache_text(text)
        action = self._clean_action_text(text)
        metadata: Dict[str, Any] = {
            "rules": [],
            "raw_action": action,
        }

        # FAQ: wifi/password
        if re.search(r"\b(wifi|wi fi|internet|mạng|pass|password|mật khẩu)\b", normalized):
            metadata["rules"].append("faq_wifi")
            if language == Language.EN:
                return "ask wifi password", "wifi password", 0.98, metadata
            return "hỏi mật khẩu wifi", "mật khẩu wifi", 0.98, metadata

        # FAQ: opening hours
        if re.search(r"(mấy giờ|giờ mở cửa|giờ đóng cửa|đóng cửa|mở cửa|open|close|closing)", normalized):
            metadata["rules"].append("faq_opening_hours")
            if language == Language.EN:
                return "ask opening hours", "opening hours", 0.96, metadata
            return "hỏi giờ mở cửa đóng cửa", "giờ mở cửa đóng cửa", 0.96, metadata

        # FAQ: payment
        if re.search(r"(thanh toán|momo|chuyển khoản|tiền mặt|visa|bank|payment|pay)", normalized):
            metadata["rules"].append("faq_payment")
            if language == Language.EN:
                return "ask payment methods", "payment methods", 0.95, metadata
            return "hỏi phương thức thanh toán", "phương thức thanh toán", 0.95, metadata

        # FAQ: delivery/takeaway
        if re.search(r"(giao hàng|ship|delivery|mang đi|take away|takeaway)", normalized):
            metadata["rules"].append("faq_delivery")
            if intent == Intent.FAQ:
                if language == Language.EN:
                    return "ask delivery takeaway policy", "delivery takeaway policy", 0.94, metadata
                return "hỏi chính sách giao hàng mang đi", "giao hàng mang đi", 0.94, metadata

        # Consultant: recommendation
        has_recommend = re.search(
            r"(gợi ý|recommend|tư vấn|món nào|có gì ngon|nên uống|dễ uống|best)",
            normalized,
        )
        has_budget = re.search(
            r"(rẻ|giá mềm|tiết kiệm|ngon rẻ|không quá mắc|budget|cheap|affordable)",
            normalized,
        )

        if intent == Intent.CONSULTANT or has_recommend:
            metadata["rules"].append("consultant_recommendation")
            if has_budget:
                metadata["rules"].append("consultant_budget")
                if language == Language.EN:
                    return (
                        "recommend affordable easy to drink item",
                        "recommend affordable easy to drink item",
                        0.94,
                        metadata,
                    )
                return (
                    "gợi ý món dễ uống giá mềm",
                    "gợi ý món ngon rẻ dễ uống",
                    0.94,
                    metadata,
                )

            if language == Language.EN:
                return "recommend menu item", "recommend menu item", 0.90, metadata
            return "gợi ý món ngon", "gợi ý món ngon", 0.90, metadata

        # Order: keep item-sensitive cache_key, but order remains no-cache in CacheService.
        if intent == Intent.ORDER:
            metadata["rules"].append("order_keep_specific")
            order_action = action
            order_action = re.sub(r"\b(cho|anh|chị|em|mình|tôi|tui|lấy|gọi|order|đặt|mua)\b", " ", order_action)
            order_action = re.sub(r"\s+", " ", order_action).strip()
            return action or normalized, order_action or normalized, 0.80, metadata

        # Ignore/noise
        if intent == Intent.IGNORE:
            metadata["rules"].append("ignore")
            return action or normalized, action or normalized, 0.75, metadata

        metadata["rules"].append("fallback")
        return action or normalized, action or normalized, 0.70, metadata

    async def extract(self, extractor_input: IntentExtractionInput) -> IntentExtractionOutput:
        language = self._detect_language(
            extractor_input.text,
            extractor_input.language,
        )

        subject = self._extract_subject(extractor_input.text, language)
        context = self._extract_context(extractor_input.text)
        action, cache_key, confidence, metadata = self._classify_action_cache_key(
            text=extractor_input.text,
            intent=extractor_input.intent,
            language=language,
        )

        metadata.update(
            {
                "extractor_backend": "rule_based",
                "router_intent": extractor_input.intent.value,
                "cache_key_source": "rule_based_action_normalization",
            }
        )

        return IntentExtractionOutput(
            subject=subject,
            action=action,
            context=context,
            cache_key=cache_key,
            intent=extractor_input.intent,
            language=language,
            confidence=confidence,
            metadata=metadata,
        )

_INTENT_EXTRACTOR_INSTANCE: BaseIntentExtractor | None = None


def get_intent_extractor() -> BaseIntentExtractor:
    global _INTENT_EXTRACTOR_INSTANCE

    if _INTENT_EXTRACTOR_INSTANCE is not None:
        return _INTENT_EXTRACTOR_INSTANCE

    settings = get_settings()
    backend = getattr(settings, "intent_extractor_backend", "rule_based")

    if backend in {"hf_lora", "hf_merged"}:
        from app.cache.intent_extractor_hf import HFIntentExtractor

        _INTENT_EXTRACTOR_INSTANCE = HFIntentExtractor()
    else:
        _INTENT_EXTRACTOR_INSTANCE = RuleBasedIntentExtractor()

    return _INTENT_EXTRACTOR_INSTANCE
