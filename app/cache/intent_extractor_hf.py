from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.cache.intent_extractor import (
    BaseIntentExtractor,
    IntentExtractionInput,
    IntentExtractionOutput,
    RuleBasedIntentExtractor,
)
from app.core.config import get_settings
from app.core.constants import Intent, Language


SYSTEM_PROMPT = (
    "You are an intent extraction model for an F&B multi-agent assistant. "
    "Extract structured fields from the customer query. "
    "Return valid JSON only. "
    "Do not add markdown. "
    "Required fields: subject, action, context, cache_key, intent, language. "
    "intent must be one of: order, consultant, faq, ignore. "
    "language must be vi or en."
)


class HFIntentExtractor(BaseIntentExtractor):
    def __init__(self) -> None:
        self.settings = get_settings()
        self.fallback = RuleBasedIntentExtractor()

        self.backend = self.settings.intent_extractor_backend
        self.base_model = self.settings.intent_extractor_base_model
        self.adapter_dir = self.settings.intent_extractor_adapter_dir
        self.merged_model_dir = self.settings.intent_extractor_merged_model_dir
        self.device = self.settings.intent_extractor_device
        self.max_new_tokens = self.settings.intent_extractor_max_new_tokens

        if self.backend == "hf_merged":
            model_path = self.merged_model_dir
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=False,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model,
                trust_remote_code=True,
                use_fast=False,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            self.model = PeftModel.from_pretrained(base_model, self.adapter_dir)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()

    def _extract_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        text = text.strip()

        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None

        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None

        return None

    def _normalize_intent(self, value: Any, router_intent: Intent) -> Intent:
        try:
            return Intent(str(value))
        except Exception:
            return router_intent

    def _normalize_language(self, value: Any, fallback: Language) -> Language:
        try:
            return Language(str(value))
        except Exception:
            return fallback

    def _is_valid_obj(self, obj: Dict[str, Any]) -> bool:
        required = {"subject", "action", "context", "cache_key", "intent", "language"}
        return required.issubset(set(obj.keys()))

    async def extract(self, extractor_input: IntentExtractionInput) -> IntentExtractionOutput:
        start = time.perf_counter()

        if extractor_input.intent in {Intent.ORDER, Intent.IGNORE}:
            output = await self.fallback.extract(extractor_input)
            output.metadata.update(
                {
                    "extractor_backend": "rule_based_fast_path",
                    "reason": "skip_hf_for_order_ignore",
                    "fallback": False,
                }
            )
            return output

        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": extractor_input.text},
            ]

            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            generated = outputs[0][inputs["input_ids"].shape[-1]:]
            raw = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

            obj = self._extract_json_object(raw)
            if obj is None or not self._is_valid_obj(obj):
                raise ValueError(f"invalid_extractor_json: {raw}")

            predicted_intent = self._normalize_intent(obj.get("intent"), extractor_input.intent)
            predicted_language = self._normalize_language(
                obj.get("language"),
                extractor_input.language,
            )

            # Router vẫn là source of truth cho dispatch/cache policy.
            router_intent = extractor_input.intent
            intent_mismatch = predicted_intent != router_intent

            latency_ms = (time.perf_counter() - start) * 1000

            return IntentExtractionOutput(
                subject=str(obj.get("subject", "") or "").strip(),
                action=str(obj.get("action", "") or "").strip(),
                context=str(obj.get("context", "") or "").strip(),
                cache_key=str(obj.get("cache_key", "") or "").strip(),
                intent=router_intent,
                language=predicted_language,
                confidence=0.90,
                metadata={
                    "extractor_backend": self.backend,
                    "base_model": self.base_model,
                    "adapter_dir": self.adapter_dir if self.backend == "hf_lora" else None,
                    "merged_model_dir": self.merged_model_dir if self.backend == "hf_merged" else None,
                    "latency_ms": latency_ms,
                    "raw_output": raw,
                    "model_intent": predicted_intent.value,
                    "router_intent": router_intent.value,
                    "intent_mismatch": intent_mismatch,
                    "fallback": False,
                },
            )

        except Exception as exc:
            fallback_output = await self.fallback.extract(extractor_input)
            fallback_output.metadata.update(
                {
                    "fallback": True,
                    "fallback_from": self.backend,
                    "fallback_error": str(exc),
                }
            )
            return fallback_output