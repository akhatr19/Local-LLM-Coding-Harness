from pathlib import Path

import pytest
from pydantic import ValidationError

from local_llm_harness.config import HarnessSettings, load_settings


def test_defaults_are_valid() -> None:
    settings = HarnessSettings()

    assert settings.litellm.default_profile == "local"
    assert settings.agents.investigator_count == 4
    assert settings.docker.network_disabled is True


def test_environment_overrides_yaml(monkeypatch, tmp_path) -> None:
    config = tmp_path / "harness.yaml"
    config.write_text("agents:\n  max_concurrency: 2\n", encoding="utf-8")
    monkeypatch.setenv("HARNESS_AGENTS__MAX_CONCURRENCY", "3")

    settings = load_settings(config)

    assert settings.agents.max_concurrency == 3


def test_unknown_default_profile_is_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown default LiteLLM profile"):
        HarnessSettings(litellm={"default_profile": "missing"})


def test_invalid_chunk_overlap_is_rejected() -> None:
    with pytest.raises(ValidationError, match="chunk_overlap_lines"):
        HarnessSettings(retrieval={"chunk_lines": 50, "chunk_overlap_lines": 50})


def test_secret_is_redacted() -> None:
    settings = HarnessSettings(
        litellm={
            "profiles": {
                "local": {
                    "model": "test/model",
                    "api_key": "super-secret",
                }
            }
        }
    )

    rendered = str(settings.redacted_dict())

    assert "super-secret" not in rendered
    assert "**********" in rendered


def test_missing_config_has_clear_error(tmp_path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        load_settings(Path(tmp_path / "missing.yaml"))
