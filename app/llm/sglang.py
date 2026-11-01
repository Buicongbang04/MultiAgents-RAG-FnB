import json
import time
from typing import Any, AsyncIterator, Dict, List

import httpx

from app.llm.base import BaseLLMClient, LLMGenerateRequest, LLMGenerateResponse, T
from app.llm.errors import LLMStructuredOutputError, LLMUnavailableError


class SGLangClient(BaseLLMClient):
    backend = "sglang"

    def __init__(self) -> None:
        from app.core.config import get_settings
        self.settings = get_settings()
        self.model = self.settings.llm.model
        self.base_url = self.settings.llm.base_url.rstrip("/")
        self.api_key = self.settings.llm.api_key
        # Một AsyncClient dùng chung (connection pool), đóng ở aclose().
        self._http = httpx.AsyncClient(timeout=self.settings.llm.timeout_seconds)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def generate_structured(self, request: LLMGenerateRequest, schema: type[T]) -> T:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": self._build_messages(request),
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema.model_json_schema(),
                    "strict": True,
                },
            },
        }
        try:
            response = await self._http.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            raise LLMUnavailableError(f"SGLang structured call failed: {exc}") from exc
        try:
            return schema.model_validate_json(content)
        except ValueError as exc:
            raise LLMStructuredOutputError(str(exc)) from exc

    def _build_messages(self, request: LLMGenerateRequest) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []

        system_prompt = request.system_prompt
        if request.summary:
            system_prompt = (
                f"{system_prompt}\n\n"
                f"TÓM TẮT HỘI THOẠI TRƯỚC ĐÓ:\n{request.summary}"
            ).strip()

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        for msg in request.history:
            if msg.role in {"user", "assistant", "system"} and msg.content.strip():
                messages.append({"role": msg.role, "content": msg.content})

        user_content = request.user_prompt
        if request.context:
            user_content = (
                "CONTEXT:\n"
                f"{request.context}\n\n"
                "USER QUESTION:\n"
                f"{request.user_prompt}"
            )

        messages.append({"role": "user", "content": user_content})
        return messages

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _fallback(self, request: LLMGenerateRequest) -> str:
        return request.metadata.get(
            "fallback_answer",
            "Hệ thống chưa phản hồi. Anh/chị vui lòng thử lại ạ.",
        )

    async def generate(self, request: LLMGenerateRequest) -> LLMGenerateResponse:
        if request.metadata.get("fallback_answer") and not request.context:
            return LLMGenerateResponse(
                text=request.metadata["fallback_answer"],
                backend=self.backend,
                model=self.model,
                metadata={"fallback_used": True, "reason": "empty_context_guardrail"},
            )

        started = time.perf_counter()
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": self._build_messages(request),
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": False,
        }

        try:
            response = await self._http.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
            )
            response.raise_for_status()
            data = response.json()
            text = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            ) or self._fallback(request)

            return LLMGenerateResponse(
                text=text,
                backend=self.backend,
                model=self.model,
                metadata={
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "fallback_used": False,
                },
            )

        except Exception as exc:
            return LLMGenerateResponse(
                text=self._fallback(request),
                backend=self.backend,
                model=self.model,
                metadata={"fallback_used": True, "error": str(exc)},
            )

    async def stream_generate(self, request: LLMGenerateRequest) -> AsyncIterator[str]:
        """
        True token streaming via SGLang SSE.
        Yields one string token at a time.
        Falls back to word-split of fallback_answer if SGLang unreachable.
        """
        if request.metadata.get("fallback_answer") and not request.context:
            for word in request.metadata["fallback_answer"].split():
                yield word + " "
            return

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": self._build_messages(request),
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": True,
        }

        try:
            async with self._http.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=self.settings.llm.stream_timeout_seconds,
            ) as response:
                response.raise_for_status()
                done = False
                async for line in response.aiter_lines():
                    if done:
                        continue
                    if not line:
                        continue
                    if line.startswith("data:"):
                        raw = line[len("data:"):].strip()
                    else:
                        raw = line.strip()

                    if raw == "[DONE]":
                        done = True
                        continue

                    try:
                        chunk = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    token = (
                        chunk.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content")
                    )
                    if token:
                        yield token

        except Exception:
            fallback = self._fallback(request)
            for word in fallback.split():
                yield word + " "
