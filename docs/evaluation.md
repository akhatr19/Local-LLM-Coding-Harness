# SWE-bench evaluation

Install the optional evaluator environment:

```sh
uv sync --locked --extra swebench
```

Run one instance through both modes before starting the full matrix:

```sh
uv run --extra swebench harness eval \
  --manifest benchmarks/swebench_lite_10.yaml \
  --config harness.local.yaml \
  --profile local \
  --mode both \
  --smoke
```

The command prints an evaluation UUID. Resume an interrupted job with the same manifest, profile,
mode, and resource settings:

```sh
uv run --extra swebench harness eval \
  --manifest benchmarks/swebench_lite_10.yaml \
  --config harness.local.yaml \
  --profile local \
  --mode both \
  --resume EVALUATION_UUID
```

Omit `--smoke` for all ten instances. The manual GitHub Actions workflow runs the same command on a
self-hosted runner.

Results are written to `.harness/evaluations/<evaluation-id>/results.json` and `comparison.md`.
Both modes receive the same model, call, token, timeout, command, and container resource limits.
Official grading uses the pinned `swebench==4.0.3` package and its Docker evaluator.
