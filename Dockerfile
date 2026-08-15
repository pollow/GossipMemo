# Resolve and install exactly the locked production dependency set in a
# disposable builder.  The runtime image only receives the resulting venv.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --locked --no-dev --extra server --no-editable

FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GOSSIPMEMO_DATABASE_PATH=/data/gossipmemo.db \
    PATH=/app/.venv/bin:$PATH

RUN groupadd --system --gid 10001 gossipmemo \
    && useradd --system --uid 10001 --gid 10001 --home-dir /home/gossipmemo --create-home gossipmemo \
    && mkdir -p /app /data \
    && chown -R gossipmemo:gossipmemo /app /data /home/gossipmemo

WORKDIR /app
COPY --from=builder --chown=gossipmemo:gossipmemo /app/.venv /app/.venv

USER gossipmemo:gossipmemo
VOLUME ["/data"]
EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8765/healthz', timeout=3).status == 200 else 1)"]

# Keep this a single process: SQLite and the in-process LLM queue require it.
CMD ["gossipmemo", "serve", "--host", "0.0.0.0", "--port", "8765"]
