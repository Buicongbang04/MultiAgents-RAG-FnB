from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings
from app.session.session_store import (
    session_store,
)
from app.middleware.rate_limit import InMemoryRateLimitMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):

    await session_store.start_background_cleanup()

    yield

    await session_store.close()


settings = get_settings()

app = FastAPI(
    title=settings.app.title,
    version=settings.app.version,
    lifespan=lifespan,
)

app.include_router(router)

app.add_middleware(
    InMemoryRateLimitMiddleware,
    max_requests=settings.rate_limit.max_requests,
    window_seconds=settings.rate_limit.window_seconds,
    exempt_path_prefixes=settings.rate_limit.exempt_path_prefixes,
)