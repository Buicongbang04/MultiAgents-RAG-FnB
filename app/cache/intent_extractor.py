from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from app.cache.exact_cache import normalize_cache_text
from app.core.config import get_lexicon, get_settings
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

    Toàn bộ pattern/từ khoá nằm ở configs/lexicon.yaml (intent_extractor).
    """

    def __init__(self) -> None:
        self.lexicon = get_lexicon().intent_extractor
        self.language_lexicon = get_lexicon().language

    def _detect_language(self, text: str, fallback: Language) -> Language:
        if fallback and fallback != Language.UNKNOWN:
            return fallback

        normalized = text.lower()
        if re.search(self.language_lexicon.vi_diacritics_pattern, normalized):
            return Language.VI
        if any(marker in normalized for marker in self.language_lexicon.intent_extractor_vi_markers):
            return Language.VI
        return Language.EN

    def _extract_subject(self, text: str, language: Language) -> str:
        normalized = text.lower().strip()

        for pattern in self.lexicon.subject_patterns:
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if match:
                if match.lastindex:
                    return match.group(1).strip()
                return "I" if language == Language.EN else ""

        if language == Language.EN and re.search(self.lexicon.english_subject_pattern, normalized):
            return "I"

        return ""

    def _extract_context(self, text: str) -> str:
        normalized = text.lower()
        contexts: List[str] = []

        for pattern in self.lexicon.context_patterns:
            for match in re.finditer(pattern, normalized, flags=re.IGNORECASE):
                value = match.group(0).strip()
                if value and value not in contexts:
                    contexts.append(value)

        return ", ".join(contexts)

    def _clean_action_text(self, text: str) -> str:
        action = normalize_cache_text(text)

        for pattern in self.lexicon.filler_patterns:
            action = re.sub(pattern, "", action, flags=re.IGNORECASE).strip()

        action = re.sub(self.lexicon.filler_words_pattern, " ", action, flags=re.IGNORECASE)
        action = re.sub(r"\s+", " ", action).strip()
        return action

    def _extract_consultant_preferences(self, normalized: str, has_budget: Any) -> List[str]:
        preferences = [
            pref.name for pref in self.lexicon.consultant_preferences
            if re.search(pref.pattern, normalized)
        ]
        # Bỏ tiêu chí bị tiêu chí cụ thể hơn thay thế ("cà phê" khi có "không cà phê").
        for dropped, when_present in self.lexicon.preference_overrides.items():
            if when_present in preferences and dropped in preferences:
                preferences.remove(dropped)
        if has_budget:
            preferences.append(self.lexicon.budget_preference)
        return preferences

    def _classify_action_cache_key(
        self,
        text: str,
        intent: Intent,
        language: Language,
    ) -> Tuple[str, str, float, Dict[str, Any]]:
        normalized = normalize_cache_text(text)
        action = self._clean_action_text(text)
        confidence = self.lexicon.confidence
        lang_key = "en" if language == Language.EN else "vi"
        metadata: Dict[str, Any] = {
            "rules": [],
            "raw_action": action,
        }

        # FAQ: rule đầu tiên khớp quyết định cache key chủ đề.
        for rule in self.lexicon.faq_rules:
            if not re.search(rule.pattern, normalized):
                continue
            metadata["rules"].append(rule.name)
            if rule.only_for_intent and rule.only_for_intent != intent.value:
                continue
            localized = rule.en if language == Language.EN else rule.vi
            return localized.action, localized.cache_key, rule.confidence, metadata

        # Consultant: recommendation
        has_recommend = re.search(self.lexicon.recommend_pattern, normalized)
        has_budget = re.search(self.lexicon.budget_pattern, normalized)

        if intent == Intent.CONSULTANT or has_recommend:
            metadata["rules"].append("consultant_recommendation")

            # Cache key phải chứa tiêu chí tư vấn: nếu mọi câu tư vấn cùng về một key
            # thì "cà phê đậm vị" và "trời nóng uống gì" sẽ trả chung một đáp án cache.
            preferences = self._extract_consultant_preferences(normalized, has_budget)
            metadata["preferences"] = preferences

            if not preferences:
                key = self.lexicon.consultant_generic[lang_key]
                return key, key, confidence.consultant_generic, metadata

            key = self.lexicon.consultant_prefix[lang_key] + " " + " ".join(preferences)
            return key, key, confidence.consultant_with_preferences, metadata

        # Order: keep item-sensitive cache_key, but order remains no-cache in CacheService.
        if intent == Intent.ORDER:
            metadata["rules"].append("order_keep_specific")
            order_action = re.sub(self.lexicon.order_strip_pattern, " ", action)
            order_action = re.sub(r"\s+", " ", order_action).strip()
            return action or normalized, order_action or normalized, confidence.order, metadata

        # Ignore/noise
        if intent == Intent.IGNORE:
            metadata["rules"].append("ignore")
            return action or normalized, action or normalized, confidence.ignore, metadata

        metadata["rules"].append("fallback")
        return action or normalized, action or normalized, confidence.fallback, metadata

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
    backend = settings.intent_extractor.backend

    if backend in {"hf_lora", "hf_merged"}:
        from app.cache.intent_extractor_hf import HFIntentExtractor

        _INTENT_EXTRACTOR_INSTANCE = HFIntentExtractor()
    else:
        _INTENT_EXTRACTOR_INSTANCE = RuleBasedIntentExtractor()

    return _INTENT_EXTRACTOR_INSTANCE
