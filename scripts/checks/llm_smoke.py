"""Smoke test thật cho backend LLM: LLM__BACKEND=anthropic uv run python -m scripts.checks.llm_smoke"""

import asyncio
import time

from pydantic import BaseModel

from app.llm import get_llm_client
from app.llm.base import LLMGenerateRequest


class Order(BaseModel):
    item: str
    quantity: int


async def main() -> None:
    llm = get_llm_client()
    print(f"backend={llm.backend} configured_model={llm.model}")

    req = LLMGenerateRequest(
        system_prompt="Bạn là nhân viên quán cà phê, trả lời ngắn gọn bằng tiếng Việt.",
        user_prompt="Chào bạn, quán có gì ngon?",
        max_tokens=1024,
        effort="low",
    )
    t = time.perf_counter()
    resp = await llm.generate(req)
    print(f"[generate] {time.perf_counter() - t:.2f}s model={resp.model} text={resp.text[:80]!r}")

    t = time.perf_counter()
    first, chunks = None, 0
    async for piece in llm.stream_generate(req):
        if first is None:
            first = time.perf_counter() - t
        chunks += 1
    print(f"[stream] ttft={first:.2f}s total={time.perf_counter() - t:.2f}s chunks={chunks}")

    sreq = LLMGenerateRequest(
        system_prompt="Trích xuất đơn hàng từ câu của khách.",
        user_prompt="Cho mình 2 ly cà phê sữa đá nhé",
        max_tokens=1024,
        effort="low",
    )
    t = time.perf_counter()
    order = await llm.generate_structured(sreq, Order)
    print(f"[structured] {time.perf_counter() - t:.2f}s result={order!r}")
    await llm.aclose()


if __name__ == "__main__":
    asyncio.run(main())
