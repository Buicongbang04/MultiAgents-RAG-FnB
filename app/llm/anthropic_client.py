"""Backend Claude (Anthropic Messages API, có thể qua gateway tương thích như OmniRoute).

- Không gửi temperature/top_p/top_k (API hiện tại từ chối); độ "suy nghĩ" điều khiển bằng
  effort trong output_config.
- System prompt gửi dạng block có cache_control để cache prompt cố định.
- Khóa API CHỈ đọc từ ANTHROPIC_API_KEY. ANTHROPIC_AUTH_TOKEN và profile `ant auth login`
  cố ý bị bỏ qua để không rò credential không liên quan sang gateway bên thứ ba.
"""

import os
import time
from typing import Any, AsyncIterator, Dict, List, Optional

import anthropic
from dotenv import dotenv_values
from pydantic import ValidationError

from app.core.config import PROJECT_ROOT, get_settings
from app.llm.base import BaseLLMClient, LLMGenerateRequest, LLMGenerateResponse, T
from app.llm.errors import (
    LLMConfigurationError,
    LLMStructuredOutputError,
    LLMUnavailableError,
)


DEFAULT_BASE_URL = "https://api.anthropic.com"  # mặc định của SDK; truyền tường minh để không đọc ANTHROPIC_BASE_URL


def _resolve_api_key() -> str:
    """Chỉ ANTHROPIC_API_KEY (env, rồi .env của dự án). Cố ý bỏ qua ANTHROPIC_AUTH_TOKEN
    và profile: không chuyển tiếp thông tin xác thực không liên quan tới gateway bên thứ ba."""
    key = os.environ.get("ANTHROPIC_API_KEY") or dotenv_values(PROJECT_ROOT / ".env").get(
        "ANTHROPIC_API_KEY"
    )
    if not key:
        raise LLMConfigurationError(
            "Backend 'anthropic' cần khóa API: đặt ANTHROPIC_API_KEY (biến môi trường hoặc .env)."
        )
    return key


class AnthropicClient(BaseLLMClient):
    backend = "anthropic"

    def __init__(self, client: Optional[Any] = None) -> None:
        cfg = get_settings().llm.anthropic
        self.model = cfg.model
        if client is not None:  # tiêm client giả cho unit test
            self._client = client
            return
        api_key = _resolve_api_key()
        kwargs: Dict[str, Any] = {
            "api_key": api_key,  # tường minh: SDK không đọc ANTHROPIC_AUTH_TOKEN
            "base_url": cfg.base_url or DEFAULT_BASE_URL,  # tường minh: SDK không đọc ANTHROPIC_BASE_URL
            "timeout": cfg.timeout_seconds,
            "max_retries": cfg.max_retries,
        }
        if cfg.base_url:
            # Gateway OmniRoute (WAF) trả 403 cho User-Agent mặc định "Anthropic/Python ...".
            kwargs["default_headers"] = {"User-Agent": "fnb-rag-backend/0.1"}
        self._client = anthropic.AsyncAnthropic(**kwargs)

    # ── request building ──────────────────────────────────────────────────────
    def _build_params(self, request: LLMGenerateRequest) -> Dict[str, Any]:
        system_text = request.system_prompt
        if request.summary:
            system_text = (
                f"{system_text}\n\nTÓM TẮT HỘI THOẠI TRƯỚC ĐÓ:\n{request.summary}"
            ).strip()

        messages: List[Dict[str, str]] = []
        for msg in request.history:
            if msg.role in {"user", "assistant"} and msg.content.strip():
                messages.append({"role": msg.role, "content": msg.content})
        while messages and messages[0]["role"] != "user":
            messages.pop(0)  # hội thoại phải bắt đầu bằng user

        user_content = request.user_prompt
        if request.context:
            user_content = (
                f"CONTEXT:\n{request.context}\n\nUSER QUESTION:\n{request.user_prompt}"
            )
        messages.append({"role": "user", "content": user_content})

        params: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_tokens,
            "messages": messages,
        }
        if system_text:
            params["system"] = [
                {
                    "type": "text",
                    "text": system_text,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        if request.effort is not None:
            params["output_config"] = {"effort": request.effort}
        return params

    @staticmethod
    def _check_stop(message: Any) -> None:
        if getattr(message, "stop_reason", None) == "refusal":
            raise LLMUnavailableError("Claude từ chối trả lời (stop_reason=refusal).")

    @staticmethod
    def _text(message: Any) -> str:
        return "".join(b.text for b in message.content if b.type == "text").strip()

    # ── API ───────────────────────────────────────────────────────────────────
    async def generate(self, request: LLMGenerateRequest) -> LLMGenerateResponse:
        started = time.perf_counter()
        try:
            message = await self._client.messages.create(**self._build_params(request))
        except anthropic.APIError as exc:
            raise LLMUnavailableError(f"Anthropic call failed: {exc}") from exc
        self._check_stop(message)
        text = self._text(message)
        if not text:
            raise LLMUnavailableError(
                f"Claude trả về rỗng (stop_reason={message.stop_reason})."
            )
        usage = getattr(message, "usage", None)
        return LLMGenerateResponse(
            text=text,
            backend=self.backend,
            model=getattr(message, "model", self.model),
            metadata={
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "stop_reason": message.stop_reason,
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
            },
        )

    async def stream_generate(self, request: LLMGenerateRequest) -> AsyncIterator[str]:
        try:
            async with self._client.messages.stream(**self._build_params(request)) as stream:
                async for text in stream.text_stream:
                    yield text
                final = await stream.get_final_message()
        except anthropic.APIError as exc:
            raise LLMUnavailableError(f"Anthropic stream failed: {exc}") from exc
        self._check_stop(final)

    async def generate_structured(self, request: LLMGenerateRequest, schema: type[T]) -> T:
        try:
            message = await self._client.messages.parse(
                **self._build_params(request), output_format=schema
            )
        except ValidationError as exc:
            raise LLMStructuredOutputError(str(exc)) from exc
        except anthropic.APIError as exc:
            raise LLMUnavailableError(f"Anthropic call failed: {exc}") from exc
        self._check_stop(message)
        parsed = getattr(message, "parsed_output", None)
        if parsed is None:
            raise LLMStructuredOutputError(
                f"Không parse được đầu ra theo {schema.__name__} "
                f"(stop_reason={message.stop_reason})."
            )
        return parsed

    async def aclose(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            await close()
