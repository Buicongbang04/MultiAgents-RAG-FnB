from functools import lru_cache

from app.core.config import get_settings
from app.llm.base import BaseLLMClient
from app.llm.mock import MockLLMClient
from app.llm.sglang import SGLangClient


@lru_cache(maxsize=1)
def get_llm_client() -> BaseLLMClient:
    settings = get_settings()
    backend = settings.llm_backend.lower()

    if backend == "mock":
        return MockLLMClient()

    if backend == "sglang":
        return SGLangClient()

    if backend == "vllm":
        raise NotImplementedError(
            "vLLM client chưa implement. Sẽ dùng fallback."
        )

    raise ValueError(f"Unsupported LLM backend: {backend}")