import pytest
from pydantic import ValidationError

from app.core.config import LLMSection, get_settings


def test_settings_load_and_forbid_unknown_keys():
    llm = get_settings().llm
    assert llm.backend in {"mock", "sglang", "vllm", "anthropic"}
    with pytest.raises(ValidationError):
        LLMSection(**llm.model_dump(), typo_key=1)
