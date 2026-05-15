import asyncio
import time
from typing import Any, Dict

import httpx


BASE_URL = "http://localhost:8001"
CHAT_URL = f"{BASE_URL}/chat"


async def clear_cache(client: httpx.AsyncClient) -> None:
    response = await client.post(f"{BASE_URL}/debug/cache/clear", timeout=30)
    response.raise_for_status()


async def chat(client: httpx.AsyncClient, session_id: str, text: str) -> Dict[str, Any]:
    start = time.perf_counter()

    response = await client.post(
        CHAT_URL,
        json={
            "session_id": session_id,
            "text": text,
        },
        timeout=120,
    )
    response.raise_for_status()

    latency_ms = (time.perf_counter() - start) * 1000
    data = response.json()
    cache = data.get("metadata", {}).get("cache", {})

    return {
        "text": text,
        "latency_ms": latency_ms,
        "intent": data.get("intent"),
        "cache_hit": cache.get("hit"),
        "cache_type": cache.get("cache_type"),
        "similarity": cache.get("similarity"),
        "matched_query": cache.get("matched_query"),
        "stored": cache.get("stored"),
        "exact_stored": (cache.get("exact") or {}).get("stored"),
        "semantic_stored": (cache.get("semantic") or {}).get("stored"),
        "answer": data.get("answer"),
        "raw_cache": cache,
    }


async def run_case(
    client: httpx.AsyncClient,
    name: str,
    seed: str,
    paraphrase: str,
) -> None:
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    await clear_cache(client)

    first = await chat(client, f"{name}-seed", seed)

    stats = await client.get(f"{BASE_URL}/debug/cache/stats", timeout=30)
    print("CACHE STATS AFTER SEED:")
    print(stats.json())

    second = await chat(client, f"{name}-para", paraphrase)

    for item in [first, second]:
        print(
            f"{item['latency_ms']:.2f}ms | "
            f"intent={item['intent']} | "
            f"hit={item['cache_hit']} | "
            f"type={item['cache_type']} | "
            f"sim={item['similarity']} | "
            f"matched={item['matched_query']} | "
            f"text={item['text']}"
        )


async def main() -> None:
    async with httpx.AsyncClient() as client:
        await run_case(
            client,
            "faq_semantic",
            "Wifi quán là gì?",
            "Cho em xin mật khẩu wifi với?",
        )

        await run_case(
            client,
            "consultant_semantic",
            "Có gì ngon rẻ không?",
            "Em gợi ý món nào giá mềm mà dễ uống?",
        )

        await clear_cache(client)

        print("\n" + "=" * 80)
        print("order_no_cache")
        print("=" * 80)

        order_1 = await chat(client, "order-clean", "Cho anh một ly bạc xỉu đá")
        order_2 = await chat(client, "order-clean", "Cho anh hai ly bạc xỉu đá")

        for item in [order_1, order_2]:
            print(
                f"{item['latency_ms']:.2f}ms | "
                f"intent={item['intent']} | "
                f"hit={item['cache_hit']} | "
                f"type={item['cache_type']} | "
                f"text={item['text']}"
            )


if __name__ == "__main__":
    asyncio.run(main())