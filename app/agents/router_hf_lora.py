from __future__ import annotations

import time
from functools import lru_cache

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.core.config import get_settings
from app.core.constants import Intent, Language
from app.core.logging import get_logger
from app.core.schemas import RouterInput, RouterOutput


logger = get_logger(__name__)

INTENTS = {
    "order": Intent.ORDER,
    "consultant": Intent.CONSULTANT,
    "faq": Intent.FAQ,
    "ignore": Intent.IGNORE,
}


SYSTEM_PROMPT = """
You are an intent classification model for a coffee shop assistant.

Classify the user message into EXACTLY one intent:

- order
- consultant
- faq
- ignore

Rules:
- order:
    direct ordering or buying menu items
- consultant:
    asking recommendation, suggestion, preference
- faq:
    asking store information, wifi, payment, opening hour, delivery
- ignore:
    greeting, nonsense, unrelated, filler

Return ONLY the intent label.
""".strip()


def parse_intent(raw_text: str) -> Intent | None:
    text = raw_text.strip().lower()

    for key, intent in INTENTS.items():
        if key in text:
            return intent

    return None


def detect_language_simple(text: str) -> Language:
    lowered = text.lower()

    vi_markers = [
        "à", "á", "ạ", "ả", "ã",
        "ă", "ắ", "ằ", "ẳ", "ẵ", "ặ",
        "â", "ấ", "ầ", "ẩ", "ẫ", "ậ",
        "đ",
        "ê", "ế", "ề", "ể", "ễ", "ệ",
        "ô", "ố", "ồ", "ổ", "ỗ", "ộ",
        "ơ", "ớ", "ờ", "ở", "ỡ", "ợ",
        "ư", "ứ", "ừ", "ử", "ữ", "ự",
        "quán", "món", "cho", "anh", "chị", "em", "mình",
    ]

    if any(marker in lowered for marker in vi_markers):
        return Language.VI

    return Language.EN


class HFRouter:
    def __init__(self, merged: bool = False) -> None:
        self.settings = get_settings()
        self.merged = merged

        if merged:
            model_path = self.settings.router_merged_model_dir
            logger.info("Loading HF merged router model=%s", model_path)
        else:
            model_path = self.settings.router_base_model
            logger.info(
                "Loading HF LoRA router base_model=%s adapter=%s",
                self.settings.router_base_model,
                self.settings.router_adapter_dir,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if self.settings.router_device == "cuda" else torch.float32

        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            device_map="auto" if self.settings.router_device == "cuda" else None,
            trust_remote_code=True,
        )

        if merged:
            self.model = base_model
        else:
            self.model = PeftModel.from_pretrained(
                base_model,
                self.settings.router_adapter_dir,
            )

        if self.settings.router_device == "cpu":
            self.model.to("cpu")

        self.model.eval()

        logger.info(
            "HF router loaded successfully router_type=%s",
            "hf_merged" if merged else "hf_lora",
        )

    def build_prompt(self, text: str) -> str:
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": text,
            },
        ]

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    async def classify(self, router_input: RouterInput) -> RouterOutput:
        start = time.perf_counter()

        prompt = self.build_prompt(router_input.text)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.settings.router_max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]

        raw_output = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        )

        intent = parse_intent(raw_output)

        if intent is None:
            raise ValueError(f"Router parse failed. raw_output={raw_output!r}")

        language = router_input.language or detect_language_simple(router_input.text)

        latency_ms = (time.perf_counter() - start) * 1000
        router_type = "hf_merged" if self.merged else "hf_lora"

        return RouterOutput(
            action=intent,
            confidence=1.0,
            language=language,
            raw_output=raw_output,
            metadata={
                "router_type": router_type,
                "base_model": self.settings.router_base_model,
                "adapter_dir": None if self.merged else self.settings.router_adapter_dir,
                "merged_model_dir": (
                    self.settings.router_merged_model_dir if self.merged else None
                ),
                "latency_ms": latency_ms,
                "required_json": {"action": intent.value},
            },
        )


@lru_cache(maxsize=1)
def get_hf_lora_router() -> HFRouter:
    return HFRouter(merged=False)


@lru_cache(maxsize=1)
def get_hf_merged_router() -> HFRouter:
    return HFRouter(merged=True)