import time
from typing import Any, Dict, List

import httpx

from app.core.config import get_settings
from app.llm.base import BaseLLMClient, LLMGenerateRequest, LLMGenerateResponse


class SGLangClient(BaseLLMClient):
    backend = "sglang"

    def __init__(self) -> None:
        self.settings = get_settings()
        self.model = self.settings.generator_model
        self.base_url = self.settings.llm_base_url.rstrip("/")
        self.api_key = self.settings.llm_api_key

    def _build_messages(self, request: LLMGenerateRequest) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []

        if request.system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": request.system_prompt,
                }
            )

        for msg in request.history:
            if msg.role in {"user", "assistant", "system"} and msg.content.strip():
                messages.append(
                    {
                        "role": msg.role,
                        "content": msg.content,
                    }
                )

        user_content = request.user_prompt

        if request.context:
            user_content = (
                "CONTEXT:\n"
                f"{request.context}\n\n"
                "USER QUESTION:\n"
                f"{request.user_prompt}"
            )

        messages.append(
            {
                "role": "user",
                "content": user_content,
            }
        )

        return messages

    async def generate(
        self,
        request: LLMGenerateRequest,
    ) -> LLMGenerateResponse:
        
        if request.metadata.get("fallback_answer") and not request.context:
            return LLMGenerateResponse(
                text=request.metadata["fallback_answer"],
                backend=self.backend,
                model=self.model,
                metadata={
                    "provider": "sglang",
                    "fallback_used": True,
                    "reason": "empty_context_guardrail",
                },
            )

        started = time.perf_counter()

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": self._build_messages(request),
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": False,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )

            response.raise_for_status()
            data = response.json()

            text = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )

            latency_ms = round((time.perf_counter() - started) * 1000, 2)

            if not text:
                text = request.metadata.get(
                    "fallback_answer",
                    "Hệ thống chưa phản hồi. Anh/chị vui lòng thử lại ạ.",
                )

            return LLMGenerateResponse(
                text=text,
                backend=self.backend,
                model=self.model,
                metadata={
                    "latency_ms": latency_ms,
                    "provider": "sglang",
                    "fallback_used": False,
                },
            )

        except Exception as exc:
            fallback = request.metadata.get(
                "fallback_answer",
                "Hệ thống chưa phản hồi. Anh/chị vui lòng thử lại ạ.",
            )

            return LLMGenerateResponse(
                text=fallback,
                backend=self.backend,
                model=self.model,
                metadata={
                    "provider": "sglang",
                    "fallback_used": True,
                    "error": str(exc),
                },
            )