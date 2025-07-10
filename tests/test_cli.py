from typer.testing import CliRunner

from local_llm_harness import __version__
from local_llm_harness.cli import app
from local_llm_harness.contracts import TaskSpec
from local_llm_harness.storage import RunStore

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
    assert "LiteLLM package" in result.stdout
    assert "SKIP" in result.stdout


def test_doctor_reports_invalid_config(tmp_path) -> None:
    config = tmp_path / "invalid.yaml"
    config.write_text("agents:\n  investigator_count: 8\n", encoding="utf-8")

    result = runner.invoke(app, ["doctor", "--config", str(config)])

    assert result.exit_code == 1
    assert "Configuration error" in result.stdout


def test_inspect_displays_persisted_run(tmp_path) -> None:
    artifact_root = tmp_path / "artifacts"
    config = tmp_path / "harness.yaml"
    config.write_text(f"artifacts:\n  root: {artifact_root}\n", encoding="utf-8")
    store = RunStore(artifact_root)
    state = store.create_run(
        TaskSpec(
            task_id="issue-1",
            repository=tmp_path,
            problem_statement="Fix it.",
        )
    )

    result = runner.invoke(app, ["inspect", str(state.run_id), "--config", str(config)])

    assert result.exit_code == 0
    assert "issue-1" in result.stdout
    assert "task.json" in result.stdout


def test_inspect_missing_run_is_an_error(tmp_path) -> None:
    config = tmp_path / "harness.yaml"
    config.write_text(f"artifacts:\n  root: {tmp_path / 'artifacts'}\n", encoding="utf-8")

    result = runner.invoke(
        app,
        ["inspect", "00000000-0000-0000-0000-000000000000", "--config", str(config)],
    )

    assert result.exit_code == 1
    assert "Unable to inspect run" in result.stdout
