from functools import lru_cache

from app.core.config import get_settings
from app.llm.base import BaseLLMClient
from app.llm.mock import MockLLMClient


@lru_cache(maxsize=1)
def get_llm_client() -> BaseLLMClient:
    settings = get_settings()

    backend = settings.generator_backend.lower()

    if backend == "mock":
        return MockLLMClient()

    elif backend == "sglang":
        raise NotImplementedError(
            "SGLang client chưa implement. "
        )

    elif backend == "vllm":
        raise NotImplementedError(
            "vLLM client chưa implement. "
        )

    raise ValueError(
        f"Unsupported generator backend: {backend}"
    )