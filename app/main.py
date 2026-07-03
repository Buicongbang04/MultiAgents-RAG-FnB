from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.core.logging import get_logger
from app.rag.neo4j_client import neo4j_client
from app.session.session_store import (
    session_store,
)
from app.middleware.rate_limit import InMemoryRateLimitMiddleware

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):

    await session_store.start_background_cleanup()

    # Best-effort: create vector indexes so the fast retrieval path works.
    # Non-fatal — retrieval falls back to full-scan cosine if this fails.
    try:
        result = neo4j_client.ensure_vector_indexes()
        logger.info("Vector indexes ensured: %s", result)
    except Exception as exc:
        logger.warning("Could not ensure vector indexes at startup: %s", exc)

    yield

    await session_store.close()
    neo4j_client.close()


app = FastAPI(
    title="MultiAgent RAG FnB",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)

app.add_middleware(
    InMemoryRateLimitMiddleware,
    max_requests=60,
    window_seconds=60,
)