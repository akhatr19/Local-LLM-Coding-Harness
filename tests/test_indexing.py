from collections.abc import Sequence
from pathlib import Path

from local_llm_harness.config import RetrievalSettings
from local_llm_harness.indexing import (
    RepositoryIndexer,
    chunk_source,
    reciprocal_rank_fusion,
)
from local_llm_harness.repository import RepositoryInspector


class FakeEmbedder:
    model_name = "fake/code-embedding"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    float(lowered.count("parser") + lowered.count("parse")),
                    float(lowered.count("readme")),
                    1.0,
                ]
            )
        return vectors


def retrieval_settings() -> RetrievalSettings:
    return RetrievalSettings(
        chunk_lines=20,
        chunk_overlap_lines=2,
        lexical_limit=10,
        vector_limit=10,
        final_limit=10,
    )


def test_python_chunking_uses_top_level_symbols() -> None:
    source = '"""module"""\n\ndef first():\n    return 1\n\nclass Second:\n    pass\n'

    chunks = chunk_source("module.py", source, chunk_lines=20, overlap_lines=2)

    symbols = {chunk.symbol for chunk in chunks}
    assert {"first", "Second"}.issubset(symbols)
    assert all(chunk.start_line <= chunk.end_line for chunk in chunks)


def test_rrf_is_deterministic_and_deduplicates_each_ranking() -> None:
    scores = reciprocal_rank_fusion([["a", "b", "a"], ["b", "c"]], k=10)

    assert scores["b"] > scores["a"]
    assert scores["a"] == 1 / 11
    assert scores["c"] == 1 / 12


def test_index_cache_and_content_invalidation(sample_repository, tmp_path: Path) -> None:
    inspector = RepositoryInspector(sample_repository)
    indexer = RepositoryIndexer(
        inspector,
        retrieval_settings(),
        tmp_path / "indexes",
        FakeEmbedder(),
    )

    first = indexer.index()
    second = indexer.index()
    parser = sample_repository / "src" / "parser.py"
    parser.write_text(parser.read_text(encoding="utf-8") + "\nNEW_VALUE = 1\n", encoding="utf-8")
    third = indexer.index()

    assert first.reused is False
    assert first.manifest.chunk_count > 0
    assert second.reused is True
    assert third.reused is False
    assert third.manifest.fingerprint != first.manifest.fingerprint


def test_hybrid_search_combines_lexical_and_vector_results(
    sample_repository, tmp_path: Path
) -> None:
    indexer = RepositoryIndexer(
        RepositoryInspector(sample_repository),
        retrieval_settings(),
        tmp_path / "indexes",
        FakeEmbedder(),
    )
    indexer.index()

    hits = indexer.hybrid_search("parse_async")

    assert hits
    assert hits[0].chunk.path == "src/parser.py"
    assert "vector" in hits[0].sources
    assert any("lexical" in hit.sources for hit in hits)
