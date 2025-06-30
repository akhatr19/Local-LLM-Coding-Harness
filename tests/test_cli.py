from typer.testing import CliRunner

from local_llm_harness import __version__
from local_llm_harness.cli import app

runner = CliRunner()


def test_help_lists_foundation_commands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "doctor" in result.stdout
    assert "version" in result.stdout


def test_version() -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_doctor_accepts_example_config() -> None:
    result = runner.invoke(app, ["doctor", "--config", "harness.example.yaml"])

    assert result.exit_code == 0
    assert "PASS" in result.stdout


def test_doctor_reports_invalid_config(tmp_path) -> None:
    config = tmp_path / "invalid.yaml"
    config.write_text("agents:\n  investigator_count: 8\n", encoding="utf-8")

    result = runner.invoke(app, ["doctor", "--config", str(config)])

    assert result.exit_code == 1
    assert "Configuration error" in result.stdout
