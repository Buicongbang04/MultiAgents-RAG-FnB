from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Literal, Optional, TypeVar

from pydantic import BaseModel, Field

from app.core.config import get_settings

T = TypeVar("T", bound=BaseModel)


class LLMMessage(BaseModel):
    role: str
    content: str


class LLMGenerateRequest(BaseModel):
    system_prompt: str
    user_prompt: str
    context: Optional[str] = None
    history: List[LLMMessage] = Field(default_factory=list)
    summary: Optional[str] = None  # tóm tắt các turn cũ đã bị cắt khỏi history
    temperature: float = Field(default_factory=lambda: get_settings().llm.temperature)
    max_tokens: int = Field(default_factory=lambda: get_settings().llm.max_tokens)
    effort: Optional[Literal["low", "medium", "high"]] = None  # chỉ backend Claude dùng
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LLMGenerateResponse(BaseModel):
    text: str
    backend: str
    model: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseLLMClient(ABC):
    backend: str
    model: str

    @abstractmethod
    async def generate(
        self,
        request: LLMGenerateRequest,
    ) -> LLMGenerateResponse:
        raise NotImplementedError

    async def stream_generate(
        self,
        request: LLMGenerateRequest,
    ) -> AsyncIterator[str]:
        response = await self.generate(request)
        yield response.text

    async def generate_structured(
        self,
        request: LLMGenerateRequest,
        schema: type[T],
    ) -> T:
        """Sinh đầu ra đúng schema Pydantic.

        Raises LLMStructuredOutputError nếu không hợp lệ,
        LLMUnavailableError nếu lỗi truyền tải/timeout.
        """
        raise NotImplementedError

    async def aclose(self) -> None:
        return None
