import asyncio

from app.llm.base import BaseLLMClient, LLMGenerateRequest, LLMGenerateResponse


class MockLLMClient(BaseLLMClient):
    backend = "mock"
    model = "mock-generator"

    async def generate(self, request: LLMGenerateRequest) -> LLMGenerateResponse:
        text = request.metadata.get("fallback_answer") or (
            "Dạ, đây là phản hồi mock từ LLM Client. "
            "Backend thật như SGLang/vLLM sẽ được thay ở phase sau."
        )
        return LLMGenerateResponse(
            text=text,
            backend=self.backend,
            model=self.model,
            metadata={"mock": True},
        )

    async def stream_generate(self, request: LLMGenerateRequest):
        text = request.metadata.get("fallback_answer") or (
            "Dạ, đây là phản hồi mock từ LLM Client. "
            "Backend thật như SGLang/vLLM sẽ được thay ở phase sau."
        )
        for word in text.split():
            yield word + " "
            await asyncio.sleep(0.01)
