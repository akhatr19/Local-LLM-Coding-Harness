import asyncio
import re
from pathlib import Path

import pytest

from local_llm_harness.config import AgentSettings
from local_llm_harness.contracts import (
    EvidenceRef,
    InvestigationReport,
    InvestigationTopic,
    InvestigationTopicSet,
    InvestigatorReport,
    RunStage,
    RunStatus,
    TaskSpec,
)
from local_llm_harness.indexing import CodeChunk, RetrievalHit
from local_llm_harness.investigation import (
    REPORT_JSON,
    REPORT_MARKDOWN,
    EvidenceValidationError,
    InvestigationWorkflow,
)
from local_llm_harness.model_gateway import FakeModelGateway, ModelResult, ModelUsage
from local_llm_harness.repository import RepositoryInspector
from local_llm_harness.storage import RunStore

PARSER_EVIDENCE = EvidenceRef(
    file_path="src/parser.py",
    start_line=3,
    end_line=6,
    excerpt=(
        "class Parser:\n"
        "    def parse(self, value: str) -> str:\n"
        "        # TODO: normalize input\n"
        "        return value"
    ),
    rationale="The synchronous parser returns its input without normalization.",
)

ASYNC_EVIDENCE = EvidenceRef(
    file_path="src/parser.py",
    start_line=8,
    end_line=9,
    excerpt="async def parse_async(value: str) -> str:\n    return value",
    rationale="The asynchronous path has equivalent behavior.",
)


class FakeRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def hybrid_search(self, query: str, *, limit: int | None = None) -> list[RetrievalHit]:
        del limit
        self.queries.append(query)
        return [
            RetrievalHit(
                chunk=CodeChunk(
                    chunk_id="parser-class",
                    path="src/parser.py",
                    start_line=3,
                    end_line=6,
                    symbol="Parser",
                    content=PARSER_EVIDENCE.excerpt,
                ),
                score=0.5,
                sources=("lexical", "vector"),
            )
        ]


def task_for(repository: Path) -> TaskSpec:
    return TaskSpec(
        task_id="parser-normalization",
        repository=repository,
        problem_statement="Normalize parser input in synchronous and asynchronous paths.",
    )


def workflow_for(sample_repository, tmp_path: Path, gateway, **agent_overrides):
    retriever = FakeRetriever()
    store = RunStore(tmp_path / "artifacts")
    settings = AgentSettings(**agent_overrides)
    workflow = InvestigationWorkflow(
        gateway=gateway,
        profile_name="local",
        inspector=RepositoryInspector(sample_repository),
        retriever=retriever,
        store=store,
        settings=settings,
    )
    return workflow, store, retriever


@pytest.mark.asyncio
async def test_investigation_fan_out_consolidation_clarification_and_resume(
    sample_repository, tmp_path: Path
) -> None:
    first_topic = {
        "topics": [
            {
                "topic_id": "parser",
                "objective": "Find parser behavior.",
                "search_terms": ["Parser", "parse"],
            }
        ]
    }
    second_topic = {
        "topics": [
            {
                "topic_id": "async-parser",
                "objective": "Resolve asynchronous parity.",
                "search_terms": ["parse_async"],
            }
        ]
    }
    gateway = FakeModelGateway(
        [
            first_topic,
            InvestigatorReport(
                topic_id="parser",
                summary="The parser returns input unchanged.",
                evidence=[PARSER_EVIDENCE],
                open_questions=["Does the async path match?"],
            ),
            InvestigationReport(
                task_id="parser-normalization",
                summary="The synchronous path is understood; async behavior remains open.",
                components=["Parser"],
                evidence=[PARSER_EVIDENCE],
                open_questions=["Does the async path match?"],
                is_clear=False,
            ),
            second_topic,
            InvestigatorReport(
                topic_id="async-parser",
                summary="The asynchronous path also returns input unchanged.",
                evidence=[ASYNC_EVIDENCE],
            ),
            InvestigationReport(
                task_id="parser-normalization",
                summary="Both parser paths require the same normalization change.",
                components=["Parser", "parse_async"],
                evidence=[PARSER_EVIDENCE, ASYNC_EVIDENCE],
                is_clear=True,
            ),
        ]
    )
    workflow, store, retriever = workflow_for(sample_repository, tmp_path, gateway)

    outcome = await workflow.run(task_for(sample_repository))

    assert outcome.rounds == 2
    assert outcome.report.is_clear is True
    assert len(gateway.calls) == 6
    assert retriever.queries == ["Parser", "parse", "parse_async"]
    state = store.get_run(outcome.run_id)
    assert state.current_stage is RunStage.PLANNING
    assert state.status is RunStatus.PENDING
    assert (
        store.read_artifact_model(outcome.run_id, REPORT_JSON, InvestigationReport)
        == outcome.report
    )
    markdown = store.runs_root / str(outcome.run_id) / REPORT_MARKDOWN
    assert "Clarity gate: passed" in markdown.read_text(encoding="utf-8")

    resumed_workflow, _, _ = workflow_for(
        sample_repository,
        tmp_path,
        FakeModelGateway([]),
    )
    resumed = await resumed_workflow.run(task_for(sample_repository), run_id=outcome.run_id)

    assert resumed.reused is True
    assert resumed.report == outcome.report


@pytest.mark.asyncio
async def test_fabricated_evidence_is_rejected(sample_repository, tmp_path: Path) -> None:
    gateway = FakeModelGateway(
        [
            {
                "topics": [
                    {
                        "topic_id": "fabricated",
                        "objective": "Inspect a claimed module.",
                        "search_terms": ["missing"],
                    }
                ]
            },
            InvestigatorReport(
                topic_id="fabricated",
                summary="A nonexistent module contains the bug.",
                evidence=[
                    EvidenceRef(
                        file_path="src/missing.py",
                        start_line=1,
                        end_line=1,
                        excerpt="fabricated = True",
                        rationale="Claimed implementation.",
                    )
                ],
            ),
        ]
    )
    workflow, store, _ = workflow_for(
        sample_repository,
        tmp_path,
        gateway,
        clarification_rounds=0,
    )

    with pytest.raises(EvidenceValidationError, match="invalid evidence location"):
        await workflow.run(task_for(sample_repository))

    state = store.get_run(next(store.runs_root.iterdir()).name)
    assert state.current_stage is RunStage.INVESTIGATION
    assert state.status is RunStatus.FAILED


class TrackingGateway:
    def __init__(self) -> None:
        self.active = 0
        self.maximum_active = 0

    async def complete(self, profile_name, messages, response_model, *, max_attempts=3):
        del profile_name, max_attempts
        if response_model is InvestigationTopicSet:
            output = InvestigationTopicSet(
                topics=[
                    InvestigationTopic(
                        topic_id=f"topic-{index}",
                        objective=f"Inspect topic {index}.",
                        search_terms=["Parser"],
                    )
                    for index in range(3)
                ]
            )
        elif response_model is InvestigatorReport:
            match = re.search(r"Topic ID: (topic-\d+)", messages[-1]["content"])
            assert match is not None
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
            await asyncio.sleep(0.02)
            self.active -= 1
            output = InvestigatorReport(
                topic_id=match.group(1),
                summary="Parser behavior located.",
                evidence=[PARSER_EVIDENCE],
            )
        else:
            output = InvestigationReport(
                task_id="parser-normalization",
                summary="The relevant behavior is understood.",
                components=["Parser"],
                evidence=[PARSER_EVIDENCE],
                is_clear=True,
            )
        return ModelResult(
            output=output,
            model="fake/tracking",
            usage=ModelUsage(),
            attempts=1,
            duration_seconds=0,
        )


@pytest.mark.asyncio
async def test_investigator_concurrency_is_bounded(sample_repository, tmp_path: Path) -> None:
    gateway = TrackingGateway()
    workflow, _, _ = workflow_for(
        sample_repository,
        tmp_path,
        gateway,
        max_concurrency=2,
        clarification_rounds=0,
    )

    await workflow.run(task_for(sample_repository))

    assert gateway.maximum_active == 2
