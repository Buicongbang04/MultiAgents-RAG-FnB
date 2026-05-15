from __future__ import annotations

import json
import time
import urllib.request


API_URL = "http://localhost:8001/chat"

WARMUP_CASES = [
    "hello em",
    "Wifi quán là gì?",
    "Cho anh một ly bạc xỉu đá",
    "Có gì ngon rẻ không?",
]


def post_chat(text: str, idx: int) -> None:
    payload = {
        "session_id": f"warmup-{idx}",
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

    with urllib.request.urlopen(req, timeout=180) as resp:
        resp.read()

    latency_ms = (time.perf_counter() - start) * 1000

    print(f"{idx}. {latency_ms:.2f} ms | {text}")


def main() -> None:
    print("=" * 80)
    print("SYSTEM WARMUP")
    print("=" * 80)

    for idx, text in enumerate(WARMUP_CASES, start=1):
        post_chat(text, idx)

    print("[DONE] Warmup completed")


if __name__ == "__main__":
    main()