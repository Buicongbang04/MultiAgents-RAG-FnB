from typing import AsyncIterator, List, Optional, Union

from pydantic import BaseModel

from app.llm.base import BaseLLMClient, LLMGenerateRequest, LLMGenerateResponse, T
from app.llm.errors import LLMUnavailableError


class ScriptedLLMClient(BaseLLMClient):
    """LLM giả cho unit test / eval offline: trả lần lượt các phản hồi đã định sẵn."""

    backend = "scripted"
    model = "scripted"

    def __init__(
        self,
        text_responses: Optional[List[str]] = None,
        structured_responses: Optional[List[Union[BaseModel, dict]]] = None,
    ) -> None:
        self.text_responses = list(text_responses or [])
        self.structured_responses = list(structured_responses or [])
        self.calls: List[LLMGenerateRequest] = []

    def _pop(self, queue: list, kind: str):
        if not queue:
            raise LLMUnavailableError(f"ScriptedLLMClient hết {kind} response.")
        return queue.pop(0)

    async def generate(self, request: LLMGenerateRequest) -> LLMGenerateResponse:
        self.calls.append(request)
        text = self._pop(self.text_responses, "text")
        return LLMGenerateResponse(text=text, backend=self.backend, model=self.model)

    async def stream_generate(self, request: LLMGenerateRequest) -> AsyncIterator[str]:
        self.calls.append(request)
        yield self._pop(self.text_responses, "text")

    async def generate_structured(self, request: LLMGenerateRequest, schema: type[T]) -> T:
        self.calls.append(request)
        item = self._pop(self.structured_responses, "structured")
        if isinstance(item, schema):
            return item
        data = item.model_dump() if isinstance(item, BaseModel) else item
        return schema.model_validate(data)
