FROM python:3.11.13-slim-bookworm@sha256:86adf8dbadc3d6e82ee5dd2c74bec2e1c2467cdad47886280501df722372d2e1

ARG UV_VERSION=0.7.19
ENV PATH="/opt/harness/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/harness

RUN python -m pip install --no-cache-dir "uv==${UV_VERSION}"

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

RUN groupadd --gid 65532 harness \
    && useradd --uid 65532 --gid 65532 --create-home harness \
    && mkdir /workspace \
    && chown 65532:65532 /workspace

USER 65532:65532
WORKDIR /workspace

ENTRYPOINT ["harness"]
CMD ["--help"]
