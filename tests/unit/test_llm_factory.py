import pytest
from pydantic import BaseModel

from app.core.config import get_settings
from app.llm.base import LLMGenerateRequest
from app.llm.errors import LLMConfigurationError, LLMUnavailableError
from app.llm.factory import get_llm_client
from app.llm.mock import MockLLMClient
from app.llm.scripted import ScriptedLLMClient


class P(BaseModel):
    sugar: str


@pytest.fixture
def fresh_settings():
    get_settings.cache_clear()
    get_llm_client.cache_clear()
    yield
    get_settings.cache_clear()
    get_llm_client.cache_clear()


@pytest.fixture
def no_dotenv_key(monkeypatch):
    monkeypatch.setattr("app.llm.anthropic_client.dotenv_values", lambda *_a, **_k: {})


def test_anthropic_backend_without_key_fails_fast(monkeypatch, fresh_settings, no_dotenv_key):
    monkeypatch.setenv("LLM__BACKEND", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMConfigurationError, match="ANTHROPIC_API_KEY"):
        get_llm_client()


def test_auth_token_alone_does_not_count(monkeypatch, fresh_settings, no_dotenv_key):
    monkeypatch.setenv("LLM__BACKEND", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "unrelated-token")
    with pytest.raises(LLMConfigurationError, match="ANTHROPIC_API_KEY"):
        get_llm_client()


def test_sdk_client_gets_explicit_key_and_base_url(monkeypatch, fresh_settings):
    monkeypatch.setenv("LLM__BACKEND", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key-from-env")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "unrelated-token")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://wrong.example.com")
    sdk = get_llm_client()._client
    assert sdk.api_key == "key-from-env"
    assert sdk.auth_token is None
    assert str(sdk.base_url).rstrip("/") == get_settings().llm.anthropic.base_url.rstrip("/")


def test_anthropic_backend_with_key_builds_client(monkeypatch, fresh_settings):
    monkeypatch.setenv("LLM__BACKEND", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    assert get_llm_client().backend == "anthropic"


async def test_scripted_client_structured_roundtrip():
    client = ScriptedLLMClient(structured_responses=[{"sugar": "it"}])
    req = LLMGenerateRequest(system_prompt="s", user_prompt="u")
    assert await client.generate_structured(req, P) == P(sugar="it")
    assert client.calls == [req]


async def test_scripted_client_text_pops_in_order():
    client = ScriptedLLMClient(text_responses=["a", "b"])
    req = LLMGenerateRequest(system_prompt="s", user_prompt="u")
    assert (await client.generate(req)).text == "a"
    assert "".join([t async for t in client.stream_generate(req)]) == "b"


async def test_mock_structured_raises_unavailable():
    req = LLMGenerateRequest(system_prompt="s", user_prompt="u")
    with pytest.raises(LLMUnavailableError):
        await MockLLMClient().generate_structured(req, P)


def test_empty_base_url_uses_sdk_default_not_env(monkeypatch, fresh_settings):
    monkeypatch.setenv("LLM__BACKEND", "anthropic")
    monkeypatch.setenv("LLM__ANTHROPIC__BASE_URL", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key-from-env")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://wrong.example.com")
    sdk = get_llm_client()._client
    assert str(sdk.base_url).rstrip("/") == "https://api.anthropic.com"
