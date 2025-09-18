# v0.1.0 checklist

- [ ] `uv lock --check` succeeds.
- [ ] `uv sync --locked` succeeds in a new checkout.
- [ ] `uv sync --locked --extra swebench` resolves SWE-bench 4.0.3.
- [ ] `uv run ruff check .` succeeds.
- [ ] `uv run ruff format --check .` succeeds.
- [ ] `uv run pytest` succeeds on Python 3.11 and 3.12.
- [ ] `harness doctor --strict` passes with local services running.
- [ ] The application image builds from `docker/app.Dockerfile`.
- [ ] A fixture issue completes without modifying its source checkout.
- [ ] The one-instance baseline/full evaluation completes.
- [ ] Dependency upload times remain no later than June 30, 2025.
- [ ] Docker references contain full SHA-256 digests.
- [ ] Commands in `docs/` match the current CLI help.
- [ ] `README.md` is absent.
