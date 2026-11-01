FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
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
COPY run.py ./

EXPOSE 8001
CMD ["uv", "run", "--frozen", "--no-dev", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
