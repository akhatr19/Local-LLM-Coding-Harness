import json

from local_llm_harness.swebench import _find_resolution


def test_find_resolution_reads_official_instance_report(tmp_path) -> None:
    report = tmp_path / "logs" / "report.json"
    report.parent.mkdir()
    report.write_text(
        json.dumps({"owner__repo-1": {"resolved": True}}),
        encoding="utf-8",
    )

    assert _find_resolution(tmp_path, "owner__repo-1") is True


def test_find_resolution_reads_summary_lists(tmp_path) -> None:
    (tmp_path / "summary.json").write_text(
        json.dumps({"unresolved_ids": ["owner__repo-1"]}),
        encoding="utf-8",
    )

    assert _find_resolution(tmp_path, "owner__repo-1") is False
