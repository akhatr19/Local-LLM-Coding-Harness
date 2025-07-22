"""Code chunking, persistent vector indexing, and hybrid retrieval."""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Iterable, Sequence
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from local_llm_harness.config import RetrievalSettings
from local_llm_harness.contracts import utc_now
from local_llm_harness.repository import (
    BinaryFileError,
    FileTooLargeError,
    RepositoryInspector,
    SearchMatch,
)


class IndexModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CodeChunk(IndexModel):
    chunk_id: str
    path: str
    start_line: int
    end_line: int
    symbol: str | None = None
    content: str


class IndexManifest(IndexModel):
    repository: str
    commit: str | None
    fingerprint: str
    config_hash: str
    embedding_model: str
    collection_name: str
    chunk_count: int = Field(ge=0)
    created_at: datetime


class IndexOutcome(IndexModel):
    manifest: IndexManifest
    reused: bool


class VectorHit(IndexModel):
    chunk: CodeChunk
    distance: float


class RetrievalHit(IndexModel):
    chunk: CodeChunk
    score: float
    sources: tuple[str, ...]
    lexical_matches: tuple[SearchMatch, ...] = ()


class EmbeddingProvider(Protocol):
    @property
    def model_name(self) -> str: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    """Lazily load the configured local embedding model."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: Any | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        vectors = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [list(map(float, vector)) for vector in vectors]


def chunk_source(
    path: str,
    source: str,
    *,
    chunk_lines: int,
    overlap_lines: int,
) -> list[CodeChunk]:
    """Chunk Python by top-level symbols and other text by bounded line windows."""

    lines = source.splitlines(keepends=True)
    if not lines:
        return []
    if path.endswith(".py"):
        try:
            tree = ast.parse(source, filename=path)
        except SyntaxError:
            tree = None
        if tree is not None:
            return _python_chunks(path, lines, tree, chunk_lines, overlap_lines)
    return _window_chunks(path, lines, 1, len(lines), chunk_lines, overlap_lines)


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], *, k: int = 60) -> dict[str, float]:
    """Merge ranked identifiers deterministically using reciprocal-rank fusion."""

    if k < 1:
        raise ValueError("RRF k must be at least 1")
    scores: dict[str, float] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, identifier in enumerate(ranking, start=1):
            if identifier in seen:
                continue
            seen.add(identifier)
            scores[identifier] = scores.get(identifier, 0.0) + 1.0 / (k + rank)
    return scores


class RepositoryIndexer:
    def __init__(
        self,
        inspector: RepositoryInspector,
        settings: RetrievalSettings,
        index_root: Path,
        embedder: EmbeddingProvider,
    ) -> None:
        self.inspector = inspector
        self.settings = settings
        self.index_root = index_root.expanduser().resolve()
        self.embedder = embedder
        repository_key = hashlib.sha256(str(inspector.root).encode()).hexdigest()[:16]
        self.repository_index = self.index_root / repository_key
        self.manifest_path = self.repository_index / "manifest.json"
        self.chunks_path = self.repository_index / "chunks.jsonl"
        self.collection_name = f"repo_{repository_key}"

    def index(self, *, force: bool = False) -> IndexOutcome:
        files = self.inspector.list_files()
        fingerprint = self._fingerprint(files)
        config_hash = self._config_hash()
        existing = self._load_manifest()
        if (
            not force
            and existing is not None
            and existing.fingerprint == fingerprint
            and existing.config_hash == config_hash
            and self.chunks_path.exists()
            and self._collection_exists()
        ):
            return IndexOutcome(manifest=existing, reused=True)

        chunks = self._build_chunks(files)
        self.repository_index.mkdir(parents=True, exist_ok=True)
        self._write_chunks(chunks)
        collection = self._replace_collection()
        for batch in _batched(chunks, 128):
            embeddings = self.embedder.embed([chunk.content for chunk in batch])
            collection.upsert(
                ids=[chunk.chunk_id for chunk in batch],
                embeddings=embeddings,
                documents=[chunk.content for chunk in batch],
                metadatas=[
                    {
                        "path": chunk.path,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "symbol": chunk.symbol or "",
                    }
                    for chunk in batch
                ],
            )

        manifest = IndexManifest(
            repository=str(self.inspector.root),
            commit=self.inspector.current_commit(),
            fingerprint=fingerprint,
            config_hash=config_hash,
            embedding_model=self.embedder.model_name,
            collection_name=self.collection_name,
            chunk_count=len(chunks),
            created_at=utc_now(),
        )
        self.manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return IndexOutcome(manifest=manifest, reused=False)

    def vector_search(self, query: str, *, limit: int | None = None) -> list[VectorHit]:
        if not query:
            raise ValueError("vector query cannot be empty")
        requested = limit or self.settings.vector_limit
        collection = self._client().get_collection(self.collection_name)
        if collection.count() == 0:
            return []
        result = collection.query(
            query_embeddings=self.embedder.embed([query]),
            n_results=min(requested, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        ids = result["ids"][0]
        documents = result["documents"][0] if result.get("documents") else []
        metadatas = result["metadatas"][0] if result.get("metadatas") else []
        distances = result["distances"][0] if result.get("distances") else []
        for chunk_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=True
        ):
            hits.append(
                VectorHit(
                    chunk=CodeChunk(
                        chunk_id=chunk_id,
                        path=str(metadata["path"]),
                        start_line=int(metadata["start_line"]),
                        end_line=int(metadata["end_line"]),
                        symbol=str(metadata.get("symbol") or "") or None,
                        content=document,
                    ),
                    distance=float(distance),
                )
            )
        return hits

    def hybrid_search(self, query: str, *, limit: int | None = None) -> list[RetrievalHit]:
        chunks = self._read_chunks()
        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        lexical = self.inspector.lexical_search(query, limit=self.settings.lexical_limit)
        lexical_by_chunk: dict[str, list[SearchMatch]] = {}
        lexical_ranking: list[str] = []
        for match in lexical:
            containing = next(
                (
                    chunk
                    for chunk in chunks
                    if chunk.path == match.path and chunk.start_line <= match.line <= chunk.end_line
                ),
                None,
            )
            if containing is None:
                continue
            lexical_by_chunk.setdefault(containing.chunk_id, []).append(match)
            if containing.chunk_id not in lexical_ranking:
                lexical_ranking.append(containing.chunk_id)

        vector = self.vector_search(query, limit=self.settings.vector_limit)
        vector_ranking = [hit.chunk.chunk_id for hit in vector]
        by_id.update({hit.chunk.chunk_id: hit.chunk for hit in vector})
        scores = reciprocal_rank_fusion([lexical_ranking, vector_ranking])
        ordered = sorted(
            scores,
            key=lambda identifier: (
                -scores[identifier],
                -len(lexical_by_chunk.get(identifier, [])),
                identifier,
            ),
        )
        requested = limit or self.settings.final_limit
        results = []
        for identifier in ordered[:requested]:
            sources = []
            if identifier in lexical_by_chunk:
                sources.append("lexical")
            if identifier in vector_ranking:
                sources.append("vector")
            results.append(
                RetrievalHit(
                    chunk=by_id[identifier],
                    score=scores[identifier],
                    sources=tuple(sources),
                    lexical_matches=tuple(lexical_by_chunk.get(identifier, [])),
                )
            )
        return results

    def _build_chunks(self, files: Sequence[str]) -> list[CodeChunk]:
        chunks: list[CodeChunk] = []
        for path in files:
            try:
                raw = self.inspector.file_bytes(path)
                source = raw.decode("utf-8")
            except (BinaryFileError, FileTooLargeError, UnicodeDecodeError):
                continue
            chunks.extend(
                chunk_source(
                    path,
                    source,
                    chunk_lines=self.settings.chunk_lines,
                    overlap_lines=self.settings.chunk_overlap_lines,
                )
            )
        return sorted(chunks, key=lambda chunk: (chunk.path, chunk.start_line, chunk.chunk_id))

    def _fingerprint(self, files: Sequence[str]) -> str:
        digest = hashlib.sha256()
        commit = self.inspector.current_commit() or "uncommitted"
        digest.update(commit.encode())
        for path in files:
            digest.update(path.encode())
            try:
                digest.update(self.inspector.file_bytes(path))
            except (BinaryFileError, FileTooLargeError):
                continue
        return digest.hexdigest()

    def _config_hash(self) -> str:
        payload = {
            "retrieval": self.settings.model_dump(mode="json"),
            "embedding_model": self.embedder.model_name,
            "format_version": 1,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _load_manifest(self) -> IndexManifest | None:
        if not self.manifest_path.exists():
            return None
        try:
            return IndexManifest.model_validate_json(self.manifest_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return None

    def _write_chunks(self, chunks: Sequence[CodeChunk]) -> None:
        with self.chunks_path.open("w", encoding="utf-8") as stream:
            for chunk in chunks:
                stream.write(chunk.model_dump_json() + "\n")

    def _read_chunks(self) -> list[CodeChunk]:
        if not self.chunks_path.exists():
            raise RuntimeError("repository index does not exist; run the index command first")
        return [
            CodeChunk.model_validate_json(line)
            for line in self.chunks_path.read_text(encoding="utf-8").splitlines()
            if line
        ]

    def _client(self):
        import chromadb
        from chromadb.config import Settings

        path = self.repository_index / "chroma"
        path.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(
            path=str(path), settings=Settings(anonymized_telemetry=False)
        )

    def _collection_exists(self) -> bool:
        try:
            self._client().get_collection(self.collection_name)
        except Exception:
            return False
        return True

    def _replace_collection(self):
        client = self._client()
        with suppress(Exception):
            client.delete_collection(self.collection_name)
        return client.create_collection(
            self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )


def _python_chunks(
    path: str,
    lines: list[str],
    tree: ast.Module,
    chunk_lines: int,
    overlap_lines: int,
) -> list[CodeChunk]:
    chunks: list[CodeChunk] = []
    covered: set[int] = set()
    symbol_types = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in tree.body:
        if not isinstance(node, symbol_types):
            continue
        start = min([node.lineno, *(decorator.lineno for decorator in node.decorator_list)])
        end = getattr(node, "end_lineno", node.lineno)
        covered.update(range(start, end + 1))
        chunks.extend(
            _window_chunks(
                path,
                lines,
                start,
                end,
                chunk_lines,
                overlap_lines,
                symbol=node.name,
            )
        )

    remainder_start: int | None = None
    for line_number in range(1, len(lines) + 2):
        uncovered = line_number <= len(lines) and line_number not in covered
        if uncovered and remainder_start is None:
            remainder_start = line_number
        elif not uncovered and remainder_start is not None:
            chunks.extend(
                _window_chunks(
                    path,
                    lines,
                    remainder_start,
                    line_number - 1,
                    chunk_lines,
                    overlap_lines,
                )
            )
            remainder_start = None
    return chunks


def _window_chunks(
    path: str,
    lines: list[str],
    start_line: int,
    end_line: int,
    chunk_lines: int,
    overlap_lines: int,
    *,
    symbol: str | None = None,
) -> list[CodeChunk]:
    if end_line < start_line:
        return []
    step = max(1, chunk_lines - overlap_lines)
    chunks = []
    window_start = start_line
    while window_start <= end_line:
        window_end = min(end_line, window_start + chunk_lines - 1)
        content = "".join(lines[window_start - 1 : window_end])
        if content.strip():
            identifier = hashlib.sha256(
                f"{path}:{window_start}:{window_end}:{content}".encode()
            ).hexdigest()
            chunks.append(
                CodeChunk(
                    chunk_id=identifier,
                    path=path,
                    start_line=window_start,
                    end_line=window_end,
                    symbol=symbol,
                    content=content,
                )
            )
        if window_end == end_line:
            break
        window_start += step
    return chunks


def _batched(items: Sequence[CodeChunk], size: int) -> Iterable[list[CodeChunk]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])
