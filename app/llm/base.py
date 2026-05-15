from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional

from pydantic import BaseModel, Field


class LLMMessage(BaseModel):
    role: str
    content: str


class LLMGenerateRequest(BaseModel):
    system_prompt: str
    user_prompt: str
    context: Optional[str] = None
    history: List[LLMMessage] = Field(default_factory=list)
    temperature: float = 0.2
    max_tokens: int = 512
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