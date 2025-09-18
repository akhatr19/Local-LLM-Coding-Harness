# Operations

## Diagnostics

`harness doctor` checks configuration, direct dependency pins, the lockfile cutoff, writable
artifact storage, required executables, package imports, and the sandbox image reference.

Use the narrower live checks while changing one service:

```sh
uv run harness doctor --config harness.local.yaml --check-services
uv run harness doctor --config harness.local.yaml --check-embeddings
uv run harness doctor --config harness.local.yaml --check-model
```

`--strict` enables all three and treats unavailable services as failures.

## Artifacts and logs

Run state is in `.harness/runs.sqlite3`. Per-run files are in `.harness/runs/`, indexes are in
`.harness/indexes/`, and evaluations are in `.harness/evaluations/`.

Structured logs are written to `.harness/logs/harness.jsonl`. Rotation is controlled by
`logging.max_bytes` and `logging.backup_count`.

Retention only applies to normal run directories and their SQLite rows. Preview it before deleting
anything:

```sh
uv run harness prune --config harness.local.yaml
uv run harness prune --config harness.local.yaml --apply
```

Completed runs older than `artifacts.retention_days`, plus completed runs beyond
`artifacts.max_completed_runs`, are selected. Failed runs are kept when
`artifacts.retain_failed_runs` is true. Active runs are never selected.

## Resource sizing

The default implementation container uses 2 CPUs, 4 GB of memory, 1 GB of writable workspace, and
a 256 MB temporary filesystem. Large projects may need 8 GB of Docker memory and more workspace
disk. Raise one limit at a time and keep command and run timeouts bounded.

Model weights, embedding caches, repository checkouts, and SWE-bench images are outside the
per-container disk limit. Keep at least 30 GB free for a small evaluation and substantially more for
the ten-instance matrix.

## Common failures

- `sandbox image is unavailable`: pull the exact image from `docker.image`; do not remove its digest.
- `source repository must be clean`: commit or stash local changes before starting a run.
- `SearXNG returned malformed JSON`: confirm `json` is enabled under `search.formats`.
- `model request failed`: verify the profile name, API base, model identifier, and endpoint logs.
- embedding download errors: populate the model cache while network access is available, then retry.
- a resumed run rejects the repository: restore the original commit recorded in `task.json`.

Use `harness inspect RUN_ID --json` for stage status and persisted errors. The implementation folder
also contains the patch, test output, changed-file inventory, and full command transcript.
