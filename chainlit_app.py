import os
import uuid
import time
from typing import Any, Dict

import chainlit as cl
import httpx


API_BASE_URL = os.getenv("CHAINLIT_API_BASE_URL", "http://localhost:8001")
CHAT_ENDPOINT = f"{API_BASE_URL}/chat"


def get_answer(payload: Dict[str, Any]) -> str:
    for key in ["answer", "response", "message", "content"]:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return str(payload)


def get_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
    metadata = payload.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


async def call_backend(user_text: str, session_id: str) -> Dict[str, Any]:
    """
    Thử nhiều schema vì endpoint /chat của repo có thể dùng:
    - {"text": ..., "session_id": ...}
    - {"message": ..., "session_id": ...}
    - {"query": ..., "session_id": ...}
    """

    candidate_payloads = [
        {"text": user_text, "session_id": session_id},
        {"message": user_text, "session_id": session_id},
        {"query": user_text, "session_id": session_id},
        {"user_message": user_text, "session_id": session_id},
    ]

    last_error = None

    async with httpx.AsyncClient(timeout=90.0) as client:
        for payload in candidate_payloads:
            response = await client.post(CHAT_ENDPOINT, json=payload)

            if response.status_code == 200:
                return response.json()

            last_error = {
                "status_code": response.status_code,
                "payload": payload,
                "body": response.text,
            }

            if response.status_code != 422:
                break

    raise RuntimeError(f"Backend request failed: {last_error}")


@cl.on_chat_start
async def on_chat_start():
    session_id = f"chainlit-{uuid.uuid4().hex[:12]}"
    cl.user_session.set("session_id", session_id)

    await cl.Message(
        content=(
            "# MultiAgents-RAG-FnB Demo\n\n"
            "UI demo cho hệ thống Multi-Agent RAG F&B.\n\n"
            "**Demo queries:**\n"
            "- Wifi quán là gì?\n"
            "- Có gì ngon rẻ không?\n"
            "- Cho anh một bạc xỉu đá size L\n"
            "- Hôm nay thời tiết thế nào?\n\n"
            f"`session_id`: `{session_id}`"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    session_id = cl.user_session.get("session_id")
    if not session_id:
        session_id = f"chainlit-{uuid.uuid4().hex[:12]}"
        cl.user_session.set("session_id", session_id)

    user_text = message.content.strip()

    msg = cl.Message(content="Đang xử lý...")
    await msg.send()

    start = time.perf_counter()

    try:
        data = await call_backend(user_text=user_text, session_id=session_id)

    except httpx.ConnectError:
        msg.content = (
            "Không kết nối được FastAPI backend.\n\n"
            "Chạy backend trước:\n"
            "```bash\n"
            "bash scripts/restart_system.sh\n"
            "```"
        )
        await msg.update()
        return

    except Exception as exc:
        msg.content = f"Lỗi khi gọi backend:\n```text\n{type(exc).__name__}: {exc}\n```"
        await msg.update()
        return

    latency_ms = (time.perf_counter() - start) * 1000

    answer = get_answer(data)
    metadata = get_metadata(data)

    intent = data.get("intent") or metadata.get("intent") or "unknown"
    agent = data.get("agent") or metadata.get("agent") or "unknown"

    cache = metadata.get("cache", {})
    extraction = metadata.get("extraction", {})

    cache_hit = cache.get("hit", False) if isinstance(cache, dict) else False
    cache_type = cache.get("type", "none") if isinstance(cache, dict) else "none"

    cache_key = ""
    if isinstance(extraction, dict):
        cache_key = extraction.get("cache_key", "")

    msg.content = (
        f"{answer}\n\n"
        "---\n"
        f"**Intent:** `{intent}`\n"
        f"**Agent:** `{agent}`\n"
        f"**Latency:** `{latency_ms:.2f} ms`\n"
        f"**Cache hit:** `{cache_hit}`\n"
        f"**Cache type:** `{cache_type}`\n"
        f"**Cache key:** `{cache_key}`\n"
        f"**Session:** `{session_id}`"
    )

    await msg.update()


# Run chainlit file : chainlit run chainlit_app.py -w --port 8501