from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from app.core.schemas import ChatRequest
from app.queueing.request_queue import queue_manager
from app.rag.neo4j_client import neo4j_client
from app.services.chat_service import chat_service
from app.streaming.sse import stream_text_as_sse

router = APIRouter()


@router.get("/health")
async def health_check():

    neo4j_ok = neo4j_client.verify_connection()

    return {
        "status": "ok",
        "neo4j": neo4j_ok,
        "queues": queue_manager.stats(),
    }


@router.post("/chat")
async def chat_endpoint(
    request: ChatRequest,
):
    result = await chat_service.chat(request)

    return JSONResponse(
        content=result.model_dump(
            mode="json",
        )
    )

@router.post("/chat/stream")
async def chat_stream_endpoint(
    request: ChatRequest,
):
    result = await chat_service.chat(request)

    return StreamingResponse(
        stream_text_as_sse(result.answer),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )