from __future__ import annotations

import json
import statistics
import time
import urllib.request


API_URL = "http://localhost:8001/chat"

TEST_CASES = [
    "Cho anh một ly bạc xỉu đá",
    "Wifi quán là gì?",
    "Có gì ngon rẻ không?",
    "Mấy giờ đóng cửa?",
    "Can I order one latte?",
    "Any good blended drink?",
    "Có thanh toán momo không?",
    "hello em",
] * 3


def post_chat(text: str, idx: int) -> dict:
    payload = {
        "session_id": f"bench-chat-{idx}",
        "text": text,
    }

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(
        API_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()

    with urllib.request.urlopen(req, timeout=120) as resp:
        body = resp.read().decode("utf-8")

    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "text": text,
        "latency_ms": latency_ms,
        "response": json.loads(body),
    }


def main() -> None:
    results = []

    print("=" * 80)
    print("CHAT API END-TO-END BENCHMARK")
    print("=" * 80)

    for idx, text in enumerate(TEST_CASES, start=1):
        result = post_chat(text, idx)
        results.append(result)

        response = result["response"]

        print(
            f"{idx:03d} | "
            f"{result['latency_ms']:.2f} ms | "
            f"intent={response.get('intent')} | "
            f"agent={response.get('agent')} | "
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


if __name__ == "__main__":
    main()