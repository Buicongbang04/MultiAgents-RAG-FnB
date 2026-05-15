from __future__ import annotations

import json
import statistics
import time
import urllib.request


API_URL = "http://localhost:8001/chat/stream"

TEST_CASES = [
    "Cho anh một ly bạc xỉu đá",
    "Wifi quán là gì?",
    "Có gì ngon rẻ không?",
    "Can I order one latte?",
    "What time do you open?",
]


def post_stream(text: str, idx: int) -> dict:
    payload = {
        "session_id": f"bench-stream-{idx}",
        "text": text,
    }

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(
        API_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )

    start = time.perf_counter()
    first_token_time = None
    chunks = []

    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()

            if not line:
                continue

            if line.startswith("data:"):
                content = line.replace("data:", "", 1).strip()

                if content == "[DONE]":
                    break

                if first_token_time is None:
                    first_token_time = time.perf_counter()

                chunks.append(content)

    total_ms = (time.perf_counter() - start) * 1000
    ttft_ms = (
        (first_token_time - start) * 1000
        if first_token_time is not None
        else None
    )

    return {
        "text": text,
        "ttft_ms": ttft_ms,
        "total_ms": total_ms,
        "chunks": chunks,
    }


def main() -> None:
    results = []

    print("=" * 80)
    print("CHAT STREAM SSE BENCHMARK")
    print("=" * 80)

    for idx, text in enumerate(TEST_CASES, start=1):
        result = post_stream(text, idx)
        results.append(result)

        print(
            f"{idx:03d} | "
            f"TTFT={result['ttft_ms']:.2f} ms | "
            f"TOTAL={result['total_ms']:.2f} ms | "
            f"chunks={len(result['chunks'])} | "
            f"{text}"
        )

    ttfts = [x["ttft_ms"] for x in results if x["ttft_ms"] is not None]
    totals = [x["total_ms"] for x in results]

    print("\nSUMMARY")
    print("-" * 80)
    print(f"Requests: {len(results)}")

    if ttfts:
        print(f"TTFT Mean:   {statistics.mean(ttfts):.2f} ms")
        print(f"TTFT Median: {statistics.median(ttfts):.2f} ms")
        print(f"TTFT P95:    {statistics.quantiles(ttfts, n=20)[18]:.2f} ms")

    print(f"Total Mean:   {statistics.mean(totals):.2f} ms")
    print(f"Total Median: {statistics.median(totals):.2f} ms")
    print(f"Total P95:    {statistics.quantiles(totals, n=20)[18]:.2f} ms")


if __name__ == "__main__":
    main()