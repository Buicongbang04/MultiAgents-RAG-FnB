"""
Chainlit UI cho MultiAgents RAG FnB.

Luồng:
1. Gọi /chat (non-streaming) để lấy full response có cấu trúc
2. Phân tích intent + nội dung câu trả lời
3. Animate từng từ với tốc độ tự nhiên → trông giống AI đang "nghĩ và gõ"
4. Hiển thị footer nhỏ gọn bên dưới
"""

import asyncio
import os
import time
import uuid
from typing import Any, Dict, Tuple

import chainlit as cl
import httpx

API_BASE_URL = os.getenv("CHAINLIT_API_BASE_URL", "http://localhost:8001")
CHAT_URL = f"{API_BASE_URL}/chat"

# ── Intent config ────────────────────────────────────────────────────────────

INTENT_META: Dict[str, Dict[str, str]] = {
    "order":      {"emoji": "🛒", "label": "Đặt hàng",  "color": "orange"},
    "consultant": {"emoji": "💡", "label": "Tư vấn",    "color": "blue"},
    "faq":        {"emoji": "❓", "label": "Hỏi đáp",   "color": "green"},
    "ignore":     {"emoji": "🔇", "label": "Không rõ",  "color": "gray"},
}

# ── Welcome ──────────────────────────────────────────────────────────────────

WELCOME = """\
## Xin chào! Tôi là trợ lý Highlands Coffee ☕

Bạn có thể hỏi tôi về:

| | Ví dụ |
|---|---|
| 🛒 **Đặt hàng** | *"Cho anh 1 bạc xỉu đá size L"* |
| 💡 **Tư vấn menu** | *"Có gì ngon rẻ không?"*, *"Gợi ý món ít ngọt"* |
| ❓ **Thông tin quán** | *"Wifi quán là gì?"*, *"Mấy giờ đóng cửa?"* |

---
*Multi-Agent RAG · Neo4j · Qwen2.5 · SGLang*
"""

# ── Backend call ─────────────────────────────────────────────────────────────

async def fetch_chat(user_text: str, session_id: str) -> Dict[str, Any]:
    payload = {"text": user_text, "session_id": session_id}
    async with httpx.AsyncClient(timeout=90.0) as client:
        resp = await client.post(CHAT_URL, json=payload)
        resp.raise_for_status()
        return resp.json()


# ── Response analysis ─────────────────────────────────────────────────────────

def parse_response(data: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Tách answer + metadata có cấu trúc từ response của backend.
    Trả về (cleaned_answer, parsed_meta).
    """
    # Extract answer
    answer = ""
    for key in ("answer", "response", "message", "content"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            answer = val.strip()
            break
    if not answer:
        answer = "Dạ, em chưa hiểu câu hỏi của anh/chị. Anh/chị thử hỏi lại nhé ạ 😊"

    # Extract metadata
    intent = str(data.get("intent", "unknown"))
    raw_meta = data.get("metadata") or {}
    cache = raw_meta.get("cache") or {}
    extraction = raw_meta.get("extraction") or {}

    cache_hit: bool = bool(cache.get("hit"))
    cache_type: str = ""
    if cache_hit:
        exact = cache.get("exact") or {}
        semantic = cache.get("semantic") or {}
        if exact.get("hit"):
            cache_type = "exact"
        elif semantic.get("hit") or cache.get("cache_type"):
            cache_type = cache.get("cache_type") or "semantic"

    meta = {
        "intent": intent,
        "latency_ms": float(data.get("latency_ms", 0)),
        "cache_hit": cache_hit,
        "cache_type": cache_type,
        "cache_key": str(extraction.get("cache_key", "")),
    }
    return answer, meta


# ── Typing animation ──────────────────────────────────────────────────────────

async def animate(msg: cl.Message, text: str) -> None:
    """
    Hiển thị text từng từ một, mô phỏng tốc độ đánh máy tự nhiên.

    - Từ bình thường: ~28ms/từ
    - Từ dài hơn hoặc sau dấu câu: chậm hơn một chút
    - Dòng mới (bullet, xuống dòng): dừng ngắn
    """
    # Split theo dòng để giữ đúng markdown newlines
    lines = text.split("\n")
    first_token = True

    for line_idx, line in enumerate(lines):
        if line_idx > 0:
            # Emit newline character
            await msg.stream_token("\n")
            await asyncio.sleep(0.06)

        words = line.split(" ")
        for w_idx, word in enumerate(words):
            if not word:
                if w_idx < len(words) - 1:
                    await msg.stream_token(" ")
                continue

            suffix = " " if w_idx < len(words) - 1 else ""
            await msg.stream_token(word + suffix)

            # Tốc độ adaptive
            if any(word.endswith(p) for p in (".", "!", "?", ":", "—")):
                delay = 0.08      # dừng sau dấu câu
            elif word.startswith(("•", "-", "*", "**")):
                delay = 0.05      # đầu bullet
            elif len(word) > 8:
                delay = 0.035     # từ dài
            else:
                delay = 0.025     # từ thường

            if first_token:
                delay = 0.01      # từ đầu tiên xuất hiện nhanh
                first_token = False

            await asyncio.sleep(delay)


# ── Footer ────────────────────────────────────────────────────────────────────

def build_footer(meta: Dict[str, Any], session_id: str) -> str:
    intent = meta.get("intent", "unknown")
    info = INTENT_META.get(intent, {"emoji": "🤖", "label": intent})
    latency = meta.get("latency_ms", 0)

    cache_part = ""
    if meta.get("cache_hit"):
        ct = meta.get("cache_type", "").capitalize() or "Cache"
        cache_part = f" · ⚡ {ct} hit"

    return (
        f"\n\n---\n"
        f"{info['emoji']} **{info['label']}**{cache_part}"
        f"  ·  `{latency:.0f} ms`"
        f"  ·  `{session_id}`"
    )


# ── Chainlit handlers ─────────────────────────────────────────────────────────

@cl.on_chat_start
async def on_chat_start():
    session_id = f"cl-{uuid.uuid4().hex[:12]}"
    cl.user_session.set("session_id", session_id)
    await cl.Message(content=WELCOME).send()


@cl.on_message
async def on_message(message: cl.Message):
    session_id: str = cl.user_session.get("session_id") or f"cl-{uuid.uuid4().hex[:12]}"
    cl.user_session.set("session_id", session_id)

    user_text = message.content.strip()
    if not user_text:
        return

    # ── 1. Fetch full response ──────────────────────────────────────────────
    async with cl.Step(name="Đang xử lý", show_input=False) as step:
        start = time.perf_counter()
        try:
            data = await fetch_chat(user_text, session_id)
        except httpx.ConnectError:
            step.output = "❌ Không kết nối được backend"
            await step.update()
            await cl.Message(
                content=(
                    "❌ **Không kết nối được backend.**\n\n"
                    "Chạy:\n```bash\npython run.py\n```"
                )
            ).send()
            return
        except Exception as exc:
            step.output = f"❌ {type(exc).__name__}"
            await step.update()
            await cl.Message(content=f"❌ Lỗi: `{exc}`").send()
            return

        latency_ms = (time.perf_counter() - start) * 1000
        answer, meta = parse_response(data)
        meta["latency_ms"] = latency_ms
        step.output = f"✅ {meta['intent']} · {latency_ms:.0f} ms"
        await step.update()

    # ── 2. Animate câu trả lời ──────────────────────────────────────────────
    msg = cl.Message(content="")
    await msg.send()

    await animate(msg, answer)

    # ── 3. Footer metadata ──────────────────────────────────────────────────
    footer = build_footer(meta, session_id)
    await msg.stream_token(footer)
    await msg.update()
