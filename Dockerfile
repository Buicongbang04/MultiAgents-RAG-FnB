FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PATH=/app/.venv/bin:$PATH \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Dependencies first for layer caching (project itself is not packaged).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --extra ml --no-dev --no-install-project

COPY app ./app
COPY configs ./configs
COPY scripts ./scripts
COPY data/mock ./data/mock
COPY run.py ./

EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
