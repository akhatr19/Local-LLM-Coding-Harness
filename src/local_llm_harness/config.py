"""Validated application configuration with environment overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


class ModelProfile(BaseModel):
    """A LiteLLM-compatible model profile."""

    model: str = "openai/qwen3-32b"
    api_base: str = "http://localhost:4000/v1"
    api_key: SecretStr | None = None
    timeout_seconds: float = Field(default=120.0, gt=0)
    max_tokens: int = Field(default=8192, gt=0)
    temperature: float = Field(default=0.1, ge=0, le=2)


class LiteLLMSettings(BaseModel):
    default_profile: str = "local"
    profiles: dict[str, ModelProfile] = Field(default_factory=lambda: {"local": ModelProfile()})

    @model_validator(mode="after")
    def default_profile_must_exist(self) -> LiteLLMSettings:
        if self.default_profile not in self.profiles:
            raise ValueError(f"unknown default LiteLLM profile: {self.default_profile}")
        return self


class SearxNGSettings(BaseModel):
    base_url: str = "http://localhost:8080"
    timeout_seconds: float = Field(default=20.0, gt=0)
    result_limit: int = Field(default=8, ge=1, le=50)
    fetch_result_limit: int = Field(default=3, ge=1, le=10)
    max_fetch_bytes: int = Field(default=200_000, ge=1_024, le=2_000_000)


class RetrievalSettings(BaseModel):
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    lexical_limit: int = Field(default=30, ge=1)
    vector_limit: int = Field(default=30, ge=1)
    final_limit: int = Field(default=20, ge=1)
    chunk_lines: int = Field(default=120, ge=20)
    chunk_overlap_lines: int = Field(default=20, ge=0)

    @model_validator(mode="after")
    def overlap_must_be_smaller_than_chunk(self) -> RetrievalSettings:
        if self.chunk_overlap_lines >= self.chunk_lines:
            raise ValueError("chunk_overlap_lines must be smaller than chunk_lines")
        return self


class AgentSettings(BaseModel):
    investigator_count: int = Field(default=4, ge=1, le=4)
    planner_count: int = Field(default=3, ge=3, le=3)
    judge_count: int = Field(default=3, ge=3, le=3)
    max_concurrency: int = Field(default=4, ge=1)
    clarification_rounds: int = Field(default=1, ge=0, le=1)


class DockerSettings(BaseModel):
    image: str = "python:3.11.13-slim-bookworm"
    cpu_limit: float = Field(default=2.0, gt=0)
    memory_mb: int = Field(default=4096, ge=256)
    pids_limit: int = Field(default=256, ge=16)
    disk_mb: int = Field(default=1024, ge=64)
    tmpfs_mb: int = Field(default=256, ge=16)
    command_timeout_seconds: int = Field(default=300, ge=1)
    run_timeout_seconds: int = Field(default=3600, ge=1)
    max_commands: int = Field(default=12, ge=1, le=100)
    max_agent_steps: int = Field(default=24, ge=1, le=200)
    network_disabled: bool = True


class ArtifactSettings(BaseModel):
    root: Path = Path(".harness")
    retain_failed_runs: bool = True


class HarnessSettings(BaseSettings):
    """Top-level settings. Environment values take precedence over YAML values."""

    model_config = SettingsConfigDict(
        env_prefix="HARNESS_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="forbid",
    )

    litellm: LiteLLMSettings = Field(default_factory=LiteLLMSettings)
    searxng: SearxNGSettings = Field(default_factory=SearxNGSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    agents: AgentSettings = Field(default_factory=AgentSettings)
    docker: DockerSettings = Field(default_factory=DockerSettings)
    artifacts: ArtifactSettings = Field(default_factory=ArtifactSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        del settings_cls
        return env_settings, init_settings, dotenv_settings, file_secret_settings

    def redacted_dict(self) -> dict[str, Any]:
        """Return settings suitable for diagnostics without exposing credentials."""

        data = self.model_dump(mode="json")
        for profile in data["litellm"]["profiles"].values():
            if profile.get("api_key") is not None:
                profile["api_key"] = "**********"
        return data


def load_settings(config_path: Path | None = None) -> HarnessSettings:
    """Load optional YAML configuration, then apply HARNESS_* environment overrides."""

    payload: dict[str, Any] = {}
    if config_path is not None:
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError(f"configuration file does not exist: {config_path}") from exc
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid YAML in {config_path}: {exc}") from exc
        if raw is not None and not isinstance(raw, dict):
            raise ValueError("configuration root must be a mapping")
        payload = raw or {}
    return HarnessSettings(**payload)
