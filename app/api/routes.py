from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.core.schemas import ChatRequest
from app.queueing.request_queue import queue_manager
from app.rag.neo4j_client import neo4j_client
from app.services.chat_service import chat_service

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