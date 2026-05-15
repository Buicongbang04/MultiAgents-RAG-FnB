from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.session.session_store import (
    session_store,
)
from app.middleware.rate_limit import InMemoryRateLimitMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):

    await session_store.start_background_cleanup()

    yield

    await session_store.close()


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