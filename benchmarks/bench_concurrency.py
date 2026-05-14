import asyncio
import statistics
import time

from app.core.schemas import ChatRequest
from app.queueing.request_queue import queue_manager
from app.services.chat_service import chat_service


TEST_CASES = [
    "Cho anh một ly bạc xỉu đá",
    "Wifi tên gì vậy?",
    "Có gì ngon ít ngọt không em?",
    "Mấy giờ đóng cửa?",
    "haha",
]


async def run_one(i: int):
    text = TEST_CASES[i % len(TEST_CASES)]

    start = time.perf_counter()

    try:
        result = await chat_service.chat(
            ChatRequest(text=text)
        )

        latency_ms = (time.perf_counter() - start) * 1000

        return {
            "ok": True,
            "text": text,
            "intent": result.intent.value,
            "agent": result.agent.value,
            "latency_ms": latency_ms,
            "sources": len(result.sources),
            "error": None,
        }

    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000

        return {
            "ok": False,
            "text": text,
            "intent": None,
            "agent": None,
            "latency_ms": latency_ms,
            "sources": 0,
            "error": str(exc),
        }


async def main():
    concurrent_requests = 5

    start = time.perf_counter()

    results = await asyncio.gather(
        *[
            run_one(i)
            for i in range(concurrent_requests)
        ]
    )

    total_ms = (time.perf_counter() - start) * 1000

    ok_results = [x for x in results if x["ok"]]
    failed_results = [x for x in results if not x["ok"]]
    latencies = [x["latency_ms"] for x in ok_results]

    print("\n" + "=" * 80)
    print("CONCURRENCY BENCH")
    print("concurrent_requests:", concurrent_requests)
    print("ok:", len(ok_results))
    print("failed:", len(failed_results))
    print("total_wall_time_ms:", round(total_ms, 2))

    if latencies:
        print("avg_latency_ms:", round(statistics.mean(latencies), 2))
        print("min_latency_ms:", round(min(latencies), 2))
        print("max_latency_ms:", round(max(latencies), 2))
        print("median_latency_ms:", round(statistics.median(latencies), 2))

    print("\nQUEUE STATS:")
    print(queue_manager.stats())

    if failed_results:
        print("\nFAILED:")
        for item in failed_results:
            print(item["text"], item["error"])


if __name__ == "__main__":
    asyncio.run(main())