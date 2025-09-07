"""Reusable full investigation, planning, research, and implementation pipeline."""

from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import UUID

from local_llm_harness.coding import CodingOutcome, CodingWorkflow
from local_llm_harness.config import HarnessSettings
from local_llm_harness.contracts import TaskSpec
from local_llm_harness.indexing import RepositoryIndexer, SentenceTransformerEmbedder
from local_llm_harness.investigation import InvestigationWorkflow
from local_llm_harness.model_gateway import LiteLLMGateway, ModelGateway
from local_llm_harness.planning import PlanningResearchWorkflow
from local_llm_harness.repository import RepositoryError, RepositoryInspector
from local_llm_harness.research import SearxNGClient
from local_llm_harness.storage import RunStore


async def run_full_pipeline(
    *,
    repository: Path,
    issue: str,
    profile: str,
    settings: HarnessSettings,
    run_id: UUID | None = None,
    gateway: ModelGateway | None = None,
    research_client: SearxNGClient | None = None,
) -> CodingOutcome:
    inspector = RepositoryInspector(repository)
    commit = inspector.current_commit()
    if commit is None:
        raise RepositoryError("full pipeline requires a Git repository with a commit")
    store = RunStore(settings.artifacts.root)
    if run_id is None:
        task_digest = hashlib.sha256(
            f"{repository.resolve()}\0{commit}\0{issue}".encode()
        ).hexdigest()[:16]
        task = TaskSpec(
            task_id=f"task-{task_digest}",
            repository=repository.resolve(),
            problem_statement=issue,
            base_commit=commit,
        )
    else:
        task = store.read_artifact_model(run_id, "task.json", TaskSpec)
        if task.repository.expanduser().resolve() != repository.resolve():
            raise ValueError("resume repository does not match the stored task")
        if task.problem_statement != issue:
            raise ValueError("resume issue does not match the stored task")
        if task.base_commit is not None and task.base_commit != commit:
            raise ValueError("resume repository no longer matches the stored base revision")

    embedder = SentenceTransformerEmbedder(settings.retrieval.embedding_model)
    indexer = RepositoryIndexer(
        inspector,
        settings.retrieval,
        settings.artifacts.root / "indexes",
        embedder,
    )
    indexer.index()
    selected_gateway = gateway or LiteLLMGateway(settings.litellm)
    investigation = await InvestigationWorkflow(
        gateway=selected_gateway,
        profile_name=profile,
        inspector=inspector,
        retriever=indexer,
        store=store,
        settings=settings.agents,
    ).run(task, run_id=run_id)

    selected_research = research_client or SearxNGClient(settings.searxng)
    try:
        planning = await PlanningResearchWorkflow(
            gateway=selected_gateway,
            profile_name=profile,
            inspector=inspector,
            research_client=selected_research,
            store=store,
            settings=settings.agents,
        ).run(task, investigation.report, run_id=investigation.run_id)
    finally:
        if research_client is None:
            await selected_research.close()

    return await CodingWorkflow(
        gateway=selected_gateway,
        profile_name=profile,
        store=store,
        settings=settings.docker,
    ).run(task, planning.final_plan, run_id=investigation.run_id)
