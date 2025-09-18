# Local setup

The harness expects Python 3.11 or 3.12, Git, ripgrep, Docker, and a LiteLLM-compatible endpoint.

Install the pinned environment from the repository root:

```sh
python3 -m pip install --user uv==0.7.19
uv sync --locked
cp harness.example.yaml harness.local.yaml
```

Start the local search service:

```sh
docker compose -f docker/compose.searxng.yaml up -d
```

Pull the sandbox image listed under `docker.image` in `harness.local.yaml`. The reference includes
the tag for readability and the digest used at runtime.

Configure a model profile in `harness.local.yaml`. `config/model-profiles.example.yaml` contains
three examples for an OpenAI-compatible endpoint on port 4000. Put API keys in environment
variables instead of the YAML file:

```sh
export HARNESS_LITELLM__PROFILES__LOCAL__API_KEY=local
```

Run the full diagnostic before the first task:

```sh
uv run harness doctor --config harness.local.yaml --strict
```

The embedding check loads the configured Sentence Transformers model. Its first run may populate
the normal Hugging Face cache.

Create a plain-text issue file and run against a clean Git checkout:

```sh
uv run harness run /path/to/repository \
  --issue-file /path/to/issue.txt \
  --config harness.local.yaml \
  --profile local
```

The source checkout is not modified. The generated patch and command transcript are stored below
`.harness/runs/<run-id>/`.

To build the application environment in a clean image:

```sh
docker build -f docker/app.Dockerfile -t local-llm-harness:0.1.0 .
docker run --rm local-llm-harness:0.1.0 version
```

Run the CLI on the host when using the coding workflow; it needs access to the local Docker daemon
to create the isolated implementation containers.
