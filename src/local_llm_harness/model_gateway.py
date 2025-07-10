"""Async LiteLLM gateway with validation, retries, and usage accounting."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from time import monotonic
from typing import Any, Generic, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from local_llm_harness.config import LiteLLMSettings, ModelProfile

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)
CompletionCallable = Callable[..., Awaitable[Any]]
Message = Mapping[str, str]


class ModelGatewayError(RuntimeError):
    """A bounded model request failed."""


class UnknownModelProfileError(ModelGatewayError):
    """The requested model profile is not configured."""


@dataclass(frozen=True)
class ModelUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class ModelResult(Generic[ResponseModel]):
    output: ResponseModel
    model: str
    usage: ModelUsage
    attempts: int
    duration_seconds: float


class ModelGateway(Protocol):
    async def complete(
        self,
        profile_name: str,
        messages: Sequence[Message],
        response_model: type[ResponseModel],
        *,
        max_attempts: int = 3,
    ) -> ModelResult[ResponseModel]: ...


class LiteLLMGateway:
    def __init__(
        self,
        settings: LiteLLMSettings,
        *,
        completion: CompletionCallable | None = None,
        retry_delay_seconds: float = 0.25,
    ) -> None:
        self.settings = settings
        self._completion_override = completion
        self.retry_delay_seconds = retry_delay_seconds
        self._usage_by_profile: dict[str, ModelUsage] = {}

    async def complete(
        self,
        profile_name: str,
        messages: Sequence[Message],
        response_model: type[ResponseModel],
        *,
        max_attempts: int = 3,
    ) -> ModelResult[ResponseModel]:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        profile = self._profile(profile_name)
        completion = self._completion()
        request = self._request(profile, messages, response_model)
        started = monotonic()
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                raw = await asyncio.wait_for(completion(**request), timeout=profile.timeout_seconds)
                output = self._validate_content(raw, response_model)
                usage = self._extract_usage(raw)
                self._record_usage(profile_name, usage)
                return ModelResult(
                    output=output,
                    model=profile.model,
                    usage=usage,
                    attempts=attempt,
                    duration_seconds=monotonic() - started,
                )
            except (TimeoutError, ValidationError, ValueError, TypeError, KeyError) as exc:
                last_error = exc
            except Exception as exc:  # Provider exceptions vary across LiteLLM backends.
                last_error = exc

            if attempt < max_attempts and self.retry_delay_seconds:
                await asyncio.sleep(self.retry_delay_seconds * (2 ** (attempt - 1)))

        raise ModelGatewayError(
            f"model request failed after {max_attempts} attempt(s): {last_error}"
        ) from last_error

    async def check_connection(self, profile_name: str | None = None) -> str:
        """Make a minimal opt-in provider call for doctor diagnostics."""

        selected = profile_name or self.settings.default_profile
        profile = self._profile(selected)
        request = self._request(
            profile,
            [{"role": "user", "content": "Reply with OK."}],
            response_model=None,
        )
        request["max_tokens"] = 1
        try:
            raw = await asyncio.wait_for(
                self._completion()(**request), timeout=profile.timeout_seconds
            )
            self._message_content(raw)
        except Exception as exc:
            raise ModelGatewayError(f"model connectivity check failed: {exc}") from exc
        return profile.model

    def usage_for(self, profile_name: str) -> ModelUsage:
        return self._usage_by_profile.get(profile_name, ModelUsage())

    def _profile(self, profile_name: str) -> ModelProfile:
        try:
            return self.settings.profiles[profile_name]
        except KeyError as exc:
            raise UnknownModelProfileError(f"unknown model profile: {profile_name}") from exc

    def _completion(self) -> CompletionCallable:
        if self._completion_override is not None:
            return self._completion_override
        from litellm import acompletion

        return acompletion

    @staticmethod
    def _request(
        profile: ModelProfile,
        messages: Sequence[Message],
        response_model: type[BaseModel] | None,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": profile.model,
            "messages": [dict(message) for message in messages],
            "api_base": profile.api_base,
            "timeout": profile.timeout_seconds,
            "max_tokens": profile.max_tokens,
            "temperature": profile.temperature,
        }
        if profile.api_key is not None:
            request["api_key"] = profile.api_key.get_secret_value()
        if response_model is not None:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": response_model.model_json_schema(),
                    "strict": True,
                },
            }
        return request

    @classmethod
    def _validate_content(cls, raw: Any, response_model: type[ResponseModel]) -> ResponseModel:
        content = cls._message_content(raw)
        if isinstance(content, str):
            return response_model.model_validate_json(content)
        if isinstance(content, dict):
            return response_model.model_validate(content)
        raise TypeError("model response content must be a JSON string or mapping")

    @staticmethod
    def _message_content(raw: Any) -> Any:
        choices = raw.get("choices") if isinstance(raw, dict) else getattr(raw, "choices", None)
        if not choices:
            raise ValueError("model response contains no choices")
        first = choices[0]
        message = (
            first.get("message") if isinstance(first, dict) else getattr(first, "message", None)
        )
        if message is None:
            raise ValueError("model response choice contains no message")
        return (
            message.get("content")
            if isinstance(message, dict)
            else getattr(message, "content", None)
        )

    @staticmethod
    def _extract_usage(raw: Any) -> ModelUsage:
        usage = raw.get("usage", {}) if isinstance(raw, dict) else getattr(raw, "usage", {})

        def value(name: str) -> int:
            raw_value = usage.get(name, 0) if isinstance(usage, dict) else getattr(usage, name, 0)
            return int(raw_value or 0)

        prompt = value("prompt_tokens")
        completion = value("completion_tokens")
        total = value("total_tokens") or prompt + completion
        return ModelUsage(prompt, completion, total)

    def _record_usage(self, profile_name: str, usage: ModelUsage) -> None:
        current = self.usage_for(profile_name)
        self._usage_by_profile[profile_name] = ModelUsage(
            prompt_tokens=current.prompt_tokens + usage.prompt_tokens,
            completion_tokens=current.completion_tokens + usage.completion_tokens,
            total_tokens=current.total_tokens + usage.total_tokens,
        )


class FakeModelGateway:
    """Deterministic queue-backed gateway for unit and workflow tests."""

    def __init__(
        self, responses: Sequence[BaseModel | Mapping[str, Any] | str | Exception]
    ) -> None:
        self._responses = deque(responses)
        self.calls: list[tuple[str, list[dict[str, str]], type[BaseModel]]] = []

    async def complete(
        self,
        profile_name: str,
        messages: Sequence[Message],
        response_model: type[ResponseModel],
        *,
        max_attempts: int = 3,
    ) -> ModelResult[ResponseModel]:
        del max_attempts
        self.calls.append((profile_name, [dict(message) for message in messages], response_model))
        if not self._responses:
            raise ModelGatewayError("fake model response queue is empty")
        response = self._responses.popleft()
        if isinstance(response, Exception):
            raise response
        if isinstance(response, response_model):
            output = response
        elif isinstance(response, str):
            output = response_model.model_validate_json(response)
        else:
            output = response_model.model_validate(response)
        return ModelResult(
            output=output,
            model=f"fake/{profile_name}",
            usage=ModelUsage(),
            attempts=1,
            duration_seconds=0,
        )


def json_response(content: Mapping[str, Any]) -> str:
    """Convenience helper for deterministic fake responses."""

    return json.dumps(content, sort_keys=True)
