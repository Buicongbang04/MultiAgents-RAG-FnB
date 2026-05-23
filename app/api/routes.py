import time
from typing import Optional

import httpx
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse, StreamingResponse

from app.cache.cache_service import cache_service
from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.schemas import ChatRequest, ErrorResponse
from app.queueing.request_queue import QueueExecutionError, QueueTimeoutError, queue_manager
from app.rag.neo4j_client import neo4j_client
from app.services.chat_service import chat_service

router = APIRouter()
logger = get_logger(__name__)


def _error_response(status: int, error: str, message: str, detail: str = "") -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content=ErrorResponse(
            error=error,
            message=message,
            metadata={"detail": detail} if detail else {},
        ).model_dump(),
    )


@router.get("/health")
async def health_check():
    settings = get_settings()

    # SGLang / vLLM health
    llm_status = "skipped"
    if settings.llm_backend in {"sglang", "vllm"}:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{settings.llm_base_url}/models")
            llm_status = "ok" if resp.status_code == 200 else f"http_{resp.status_code}"
        except Exception as exc:
            llm_status = f"error: {exc}"

    return {
        "status": "ok",
        "neo4j": neo4j_client.verify_connection(),
        "llm": llm_status,
        "queues": queue_manager.stats(),
    }


@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        result = await chat_service.chat(request)
        return JSONResponse(content=result.model_dump(mode="json"))
    except QueueTimeoutError as exc:
        logger.warning("Queue timeout /chat: %s", exc)
        return _error_response(503, "queue_timeout", "Request chờ quá lâu. Vui lòng thử lại.", str(exc))
    except QueueExecutionError as exc:
        logger.warning("Queue error /chat: %s", exc)
        return _error_response(503, "service_unavailable", "Hệ thống tạm thời quá tải.", str(exc))
    except Exception as exc:
        logger.exception("Unexpected error /chat")
        return _error_response(500, "internal_error", "Lỗi hệ thống.", str(exc))


@router.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    True streaming endpoint.
    SSE events: METADATA (ttft_ms) → TOKEN* → CLAUSE* → [DONE]
    """
    start_time = time.perf_counter()
    return StreamingResponse(
        chat_service.chat_stream(request, start_time=start_time),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/debug/cache/stats")
async def cache_stats():
    return cache_service.stats()


@router.post("/debug/cache/clear")
async def clear_cache():
    return cache_service.clear_all()


@router.post("/cache/invalidate")
async def invalidate_cache(
    intent: Optional[str] = Query(
        default=None,
        description="Invalidate entries for a specific intent: faq | consultant | ignore. Omit to clear all.",
    )
):
    """
    Xóa cache khi menu hoặc FAQ thay đổi.
    - intent=faq    → xóa toàn bộ entries FAQ
    - intent=consultant → xóa entries Consultant
    - (không truyền) → xóa toàn bộ cache
    """
    if intent:
        result = cache_service.invalidate_by_intent(intent)
    else:
        result = cache_service.clear_all()
    return result
