import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.llm import get_llm_client
from app.rag.embedding_client import get_embedding_client
from app.rag.reranker import get_reranker
from app.session.session_store import (
    session_store,
)
from app.middleware.rate_limit import InMemoryRateLimitMiddleware

logger = get_logger(__name__)


async def _warmup_models() -> None:
    started = time.perf_counter()
    try:
        await get_embedding_client().embed_text("khởi động")
        await asyncio.to_thread(get_reranker().warmup)
    except Exception as exc:
        # Degrade: vẫn phục vụ, model sẽ nạp lười ở request đầu.
        logger.warning("Model warmup failed (%s); continuing, models will lazy-load.", exc)
        return
    logger.info("Model warmup done in %.2fs", time.perf_counter() - started)


@asynccontextmanager
async def lifespan(app: FastAPI):

    setup_logging()  # nếu không, log INFO (kể cả thời gian warmup) bị nuốt khi chạy bằng uvicorn
    llm_client = get_llm_client()  # fail-fast nếu thiếu cấu hình/khóa
    if get_settings().server.warmup_on_startup:
        await _warmup_models()
    await session_store.start_background_cleanup()

    yield

    await session_store.close()
    await llm_client.aclose()


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