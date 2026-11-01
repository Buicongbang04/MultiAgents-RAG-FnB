from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from app.core.config import get_settings
from app.llm.anthropic_client import AnthropicClient
from app.llm.base import LLMGenerateRequest
from app.llm.errors import LLMStructuredOutputError, LLMUnavailableError


def _msg(text="xin chao", stop_reason="end_turn"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)] if text else [],
        stop_reason=stop_reason,
        model="claude-opus-5-5",
        usage=SimpleNamespace(input_tokens=1, output_tokens=2),
    )


class FakeMessages:
    def __init__(self, msg):
        self.msg = msg
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return self.msg

    async def parse(self, **kwargs):
        self.kwargs = kwargs
        return self.msg


class FakeClient:
    def __init__(self, msg):
        self.messages = FakeMessages(msg)


def _client(msg):
    fake = FakeClient(msg)
    return AnthropicClient(client=fake), fake


async def test_request_has_no_temperature_and_maps_effort():
    client, fake = _client(_msg())
    req = LLMGenerateRequest(
        system_prompt="sys", user_prompt="hi", summary="old", effort="low", temperature=0.9
    )
    resp = await client.generate(req)
    kwargs = fake.messages.kwargs
    assert "temperature" not in kwargs and "top_p" not in kwargs
    assert kwargs["output_config"]["effort"] == "low"
    assert kwargs["model"] == get_settings().llm.anthropic.model
    assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "old" in kwargs["system"][0]["text"]
    assert resp.text == "xin chao" and resp.backend == "anthropic"
    assert resp.model == "claude-opus-5-5"


async def test_effort_omitted_when_none():
    client, fake = _client(_msg())
    await client.generate(LLMGenerateRequest(system_prompt="s", user_prompt="u"))
    assert "output_config" not in fake.messages.kwargs


async def test_refusal_raises_unavailable():
    client, _ = _client(_msg(text="", stop_reason="refusal"))
    with pytest.raises(LLMUnavailableError):
        await client.generate(LLMGenerateRequest(system_prompt="s", user_prompt="u"))


class P(BaseModel):
    sugar: str


async def test_structured_parsed_output_and_invalid():
    msg = _msg()
    msg.parsed_output = P(sugar="it")
    client, fake = _client(msg)
    req = LLMGenerateRequest(system_prompt="s", user_prompt="u")
    assert await client.generate_structured(req, P) == P(sugar="it")
    assert fake.messages.kwargs["output_format"] is P

    bad = _msg()
    bad.parsed_output = None
    client, _ = _client(bad)
    with pytest.raises(LLMStructuredOutputError):
        await client.generate_structured(req, P)
