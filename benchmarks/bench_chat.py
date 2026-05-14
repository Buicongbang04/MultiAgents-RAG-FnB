import asyncio
import statistics
import time

from app.core.schemas import ChatRequest
from app.services.chat_service import chat_service


TEST_CASES = [
    "Cho anh một ly bạc xỉu đá",
    "Wifi tên gì vậy?",
    "Có gì ngon ít ngọt không em?",
    "Mấy giờ đóng cửa?",
    "haha",
]


async def run_once(text: str):
    start = time.perf_counter()

    result = await chat_service.chat(
        ChatRequest(text=text)
    )

    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "text": text,
        "intent": result.intent.value,
        "agent": result.agent.value,
        "latency_ms": latency_ms,
        "sources": len(result.sources),
    }


async def main():
    results = []

    for text in TEST_CASES:
        item = await run_once(text)
        results.append(item)

        print("\n" + "=" * 80)
        print("TEXT:", item["text"])
        print("INTENT:", item["intent"])
        print("AGENT:", item["agent"])
        print("LATENCY:", round(item["latency_ms"], 2), "ms")
        print("SOURCES:", item["sources"])

    latencies = [x["latency_ms"] for x in results]

    print("\n" + "=" * 80)
    print("BENCH SUMMARY")
    print("count:", len(latencies))
    print("avg_ms:", round(statistics.mean(latencies), 2))
    print("min_ms:", round(min(latencies), 2))
    print("max_ms:", round(max(latencies), 2))

    if len(latencies) >= 2:
        print("median_ms:", round(statistics.median(latencies), 2))


if __name__ == "__main__":
    asyncio.run(main())