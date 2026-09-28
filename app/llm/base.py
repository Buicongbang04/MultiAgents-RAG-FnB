from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional

from pydantic import BaseModel, Field

from app.core.config import get_settings


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