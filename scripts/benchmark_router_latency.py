from __future__ import annotations

import asyncio
import statistics
import time
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.router_agent import router_agent
from app.core.schemas import RouterInput


TEST_CASES = [
    "Cho anh một ly bạc xỉu đá",
    "Có gì ngon rẻ không?",
    "Wifi quán là gì?",
    "hello em",
    "Can I order one latte?",
    "What time do you open?",
    "Any good blended drink?",
    "hmm not sure",
    "Có thanh toán momo không?",
    "Mình muốn gọi một latte nhé",
] * 5


async def classify_once(idx: int, text: str) -> dict:
    start = time.perf_counter()

    output = await router_agent.classify(
        RouterInput(
            session_id=f"bench-router-{idx}",
            text=text,
        )
    )

    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "idx": idx,
        "text": text,
        "intent": output.action.value,
        "latency_ms": latency_ms,
        "router_type": output.metadata.get("router_type"),
        "json": output.to_required_json(),
    }


async def main() -> None:
    print("=" * 80)
    print("ROUTER LATENCY BENCHMARK")
    print("=" * 80)

    print("[WARMUP]")
    await classify_once(0, "Wifi quán là gì?")

    results = []

    for idx, text in enumerate(TEST_CASES, start=1):
        result = await classify_once(idx, text)
        results.append(result)

        print(
            f"{idx:03d} | "
            f"{result['latency_ms']:.2f} ms | "
            f"{result['router_type']} | "
            f"{result['intent']} | "
            f"{text}"
        )

    latencies = [x["latency_ms"] for x in results]

    print("\nSUMMARY")
    print("-" * 80)
    print(f"Requests: {len(results)}")
    print(f"Mean:     {statistics.mean(latencies):.2f} ms")
    print(f"Median:   {statistics.median(latencies):.2f} ms")
    print(f"Min:      {min(latencies):.2f} ms")
    print(f"Max:      {max(latencies):.2f} ms")
    print(f"P90:      {statistics.quantiles(latencies, n=10)[8]:.2f} ms")
    print(f"P95:      {statistics.quantiles(latencies, n=20)[18]:.2f} ms")

    valid_json = all(
        isinstance(x["json"], dict)
        and x["json"].get("action") in {"order", "consultant", "faq", "ignore"}
        for x in results
    )

    print(f"JSON valid: {valid_json}")


if __name__ == "__main__":
    asyncio.run(main())