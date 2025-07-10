import asyncio
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from local_llm_harness.config import LiteLLMSettings
from local_llm_harness.model_gateway import (
    FakeModelGateway,
    LiteLLMGateway,
    ModelGatewayError,
    UnknownModelProfileError,
)


class Answer(BaseModel):
    value: str


def raw_response(content: str, *, prompt: int = 0, completion: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        ),
    )


@pytest.mark.asyncio
async def test_structured_response_and_usage_are_recorded() -> None:
    requests = []

    async def completion(**kwargs):
        requests.append(kwargs)
        return raw_response('{"value":"ok"}', prompt=5, completion=2)

    settings = LiteLLMSettings(profiles={"local": {"model": "test/model", "api_key": "secret"}})
    gateway = LiteLLMGateway(settings, completion=completion, retry_delay_seconds=0)

    result = await gateway.complete("local", [{"role": "user", "content": "answer"}], Answer)

    assert result.output == Answer(value="ok")
    assert result.usage.total_tokens == 7
    assert gateway.usage_for("local").prompt_tokens == 5
    assert requests[0]["api_key"] == "secret"
    assert requests[0]["response_format"]["type"] == "json_schema"


@pytest.mark.asyncio
async def test_malformed_response_is_retried() -> None:
    responses = iter([raw_response("not-json"), raw_response('{"value":"recovered"}')])

    async def completion(**kwargs):
        del kwargs
        return next(responses)

    gateway = LiteLLMGateway(LiteLLMSettings(), completion=completion, retry_delay_seconds=0)

    result = await gateway.complete("local", [], Answer, max_attempts=2)

    assert result.output.value == "recovered"
    assert result.attempts == 2


@pytest.mark.asyncio
async def test_timeout_is_bounded_and_reported() -> None:
    async def completion(**kwargs):
        del kwargs
        await asyncio.sleep(0.02)
        return raw_response('{"value":"late"}')

    settings = LiteLLMSettings(
        profiles={"local": {"model": "test/model", "timeout_seconds": 0.001}}
    )
    gateway = LiteLLMGateway(settings, completion=completion, retry_delay_seconds=0)

    with pytest.raises(ModelGatewayError, match="after 2 attempt"):
        await gateway.complete("local", [], Answer, max_attempts=2)


@pytest.mark.asyncio
async def test_unknown_profile_fails_before_provider_call() -> None:
    called = False

    async def completion(**kwargs):
        nonlocal called
        called = True
        return raw_response('{"value":"unexpected"}')

    gateway = LiteLLMGateway(LiteLLMSettings(), completion=completion)

    with pytest.raises(UnknownModelProfileError):
        await gateway.complete("missing", [], Answer)
    assert called is False


@pytest.mark.asyncio
async def test_fake_gateway_is_deterministic_and_never_calls_litellm() -> None:
    gateway = FakeModelGateway([{"value": "first"}, Answer(value="second")])

    first = await gateway.complete("local", [], Answer)
    second = await gateway.complete("local", [], Answer)

    assert first.output.value == "first"
    assert second.output.value == "second"
    assert len(gateway.calls) == 2
    with pytest.raises(ModelGatewayError, match="queue is empty"):
        await gateway.complete("local", [], Answer)
