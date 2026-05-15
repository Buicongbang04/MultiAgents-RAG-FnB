# 500 test case for benchmarking intelligent cache performance
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests

BASE_URL = "http://localhost:8001"
CHAT_URL = f"{BASE_URL}/chat"
CACHE_CLEAR_URL = f"{BASE_URL}/debug/cache/clear"

REPORT_DIR = Path("reports")
REPORT_PATH = REPORT_DIR / "intelligent_cache_500_benchmark.json"

TIMEOUT = 120


SEED_CASES = [
    # FAQ
    ("seed-faq-wifi", "Wifi quán là gì?"),
    ("seed-faq-hours", "Quán mở cửa mấy giờ?"),
    ("seed-faq-payment", "Có thanh toán momo không?"),
    ("seed-faq-delivery", "Có giao hàng không?"),

    # Consultant
    ("seed-consult-budget", "Có gì ngon rẻ không?"),
    ("seed-consult-low-sugar", "Tôi thích ít ngọt thì uống gì?"),
    ("seed-consult-hot-weather", "Có gì hợp trời nóng không?"),
]


FAQ_VARIANTS = [
    ("faq", "Cho em xin mật khẩu wifi với nha", True),
    ("faq", "Dạ pass wifi ở đây là gì vậy ạ?", True),
    ("faq", "Internet ở quán dùng sao em?", True),
    ("faq", "Can I get the wifi password?", True),
    ("faq", "What is the wifi password?", True),

    ("faq", "Mấy giờ đóng cửa vậy em?", True),
    ("faq", "Hôm nay quán còn mở không?", True),
    ("faq", "Bên mình mấy giờ nghỉ?", True),
    ("faq", "What time do you close?", True),
    ("faq", "What are your opening hours?", True),

    ("faq", "Quán có nhận thẻ không?", True),
    ("faq", "Em chuyển khoản được không?", True),
    ("faq", "Ở đây thanh toán bằng gì được?", True),
    ("faq", "Can I pay by card?", True),
    ("faq", "What payment methods do you accept?", True),

    ("faq", "Bên mình có ship không?", True),
    ("faq", "Có cho mang đi không?", True),
    ("faq", "Đặt take away được không?", True),
    ("faq", "Do you deliver?", True),
    ("faq", "Can I order takeaway?", True),
]

CONSULTANT_VARIANTS = [
    ("consultant", "Em gợi ý món nào giá mềm mà dễ uống?", True),
    ("consultant", "Có món nào ngon mà không quá mắc không?", True),
    ("consultant", "Tư vấn giúp mình món dễ uống giá ổn với", True),
    ("consultant", "Budget friendly mà dễ uống thì món nào ổn?", True),
    ("consultant", "Can you recommend something affordable and easy to drink?", True),
    ("consultant", "Suggest something budget friendly and easy to drink", True),

    ("consultant", "Có món nào ít ngọt dễ uống không?", True),
    ("consultant", "Gợi ý món ít ngọt cho mình", True),
    ("consultant", "Em muốn món nào ít ngọt", True),
    ("consultant", "Can you recommend something not too sweet?", True),
    ("consultant", "Any low sugar drink?", True),

    ("consultant", "Trời nóng nên uống món nào?", True),
    ("consultant", "Gợi ý món mát mát cho hôm nay đi", True),
    ("consultant", "Em muốn món mát dễ uống cho trời nóng", True),
    ("consultant", "Any drink for hot weather?", True),
    ("consultant", "Recommend something refreshing for hot weather", True),
]

ORDER_VARIANTS = [
    ("order", "Cho anh 1 bạc xỉu đá", False),
    ("order", "Cho anh 2 bạc xỉu đá", False),
    ("order", "Cho em một ly cà phê sữa đá", False),
    ("order", "Cho chị hai ly trà đào", False),
    ("order", "Can I order one latte?", False),
    ("order", "I want two cappuccinos", False),
]

IGNORE_VARIANTS = [
    ("ignore", "hello em", True),
    ("ignore", "ừm để anh xem", True),
    ("ignore", "khoan đã", True),
    ("ignore", "haha", True),
    ("ignore", "ok", True),
]


def post_chat(session_id: str, text: str) -> Tuple[Dict[str, Any], float]:
    start = time.perf_counter()
    resp = requests.post(
        CHAT_URL,
        json={"session_id": session_id, "text": text},
        timeout=TIMEOUT,
    )
    latency_ms = (time.perf_counter() - start) * 1000
    resp.raise_for_status()
    return resp.json(), latency_ms


def clear_cache() -> None:
    try:
        requests.post(CACHE_CLEAR_URL, timeout=TIMEOUT)
    except Exception as exc:
        print(f"[WARN] Could not clear cache: {exc}")


def cache_hit(payload: Dict[str, Any]) -> bool:
    return bool(payload.get("metadata", {}).get("cache", {}).get("hit", False))


def cache_type(payload: Dict[str, Any]) -> str:
    return str(payload.get("metadata", {}).get("cache", {}).get("cache_type", ""))


def generator_completed(payload: Dict[str, Any]) -> int:
    return int(
        payload.get("metadata", {})
        .get("queue_stats", {})
        .get("generator", {})
        .get("completed", -1)
    )


def extraction_backend(payload: Dict[str, Any]) -> str:
    return str(
        payload.get("metadata", {})
        .get("extraction", {})
        .get("metadata", {})
        .get("extractor_backend", "")
    )


def cache_lookup_text(payload: Dict[str, Any]) -> str:
    return str(payload.get("metadata", {}).get("cache_lookup_text", ""))


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int(round((len(sorted_values) - 1) * p))
    return sorted_values[index]


def summarize_latencies(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p90_ms": 0.0,
            "p95_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
        }

    return {
        "count": len(values),
        "mean_ms": round(statistics.mean(values), 2),
        "median_ms": round(statistics.median(values), 2),
        "p90_ms": round(percentile(values, 0.90), 2),
        "p95_ms": round(percentile(values, 0.95), 2),
        "min_ms": round(min(values), 2),
        "max_ms": round(max(values), 2),
    }


def build_500_cases() -> List[Tuple[str, str, bool]]:
    base = FAQ_VARIANTS + CONSULTANT_VARIANTS + ORDER_VARIANTS + IGNORE_VARIANTS

    cases: List[Tuple[str, str, bool]] = []
    i = 0
    while len(cases) < 500:
        intent, text, expected_cacheable = base[i % len(base)]
        # giữ text tự nhiên nhưng session riêng để tránh history gây nhiễu
        cases.append((intent, text, expected_cacheable))
        i += 1

    return cases


def main() -> None:
    print("=" * 80)
    print("INTELLIGENT CACHE 500 BENCHMARK")
    print("=" * 80)
    print(f"CHAT_URL: {CHAT_URL}")

    clear_cache()

    seed_results = []
    print("\n[1] Seeding cache...")
    for session_id, text in SEED_CASES:
        payload, latency_ms = post_chat(session_id, text)
        seed_results.append(
            {
                "session_id": session_id,
                "text": text,
                "intent": payload.get("intent"),
                "latency_ms": round(latency_ms, 2),
                "cache_hit": cache_hit(payload),
                "cache_lookup_text": cache_lookup_text(payload),
                "extractor_backend": extraction_backend(payload),
            }
        )
        print(
            f"SEED | {latency_ms:8.2f} ms | "
            f"intent={payload.get('intent')} | hit={cache_hit(payload)} | {text}"
        )

    cases = build_500_cases()

    rows = []
    hit_latencies = []
    miss_latencies = []
    all_latencies = []

    expected_cacheable_total = 0
    expected_cacheable_hit = 0

    order_total = 0
    order_false_hit = 0

    generator_bypass_hits = 0
    total_hits = 0

    before_generator_completed = None

    print("\n[2] Running 500 benchmark queries...")
    for idx, (expected_intent, text, expected_cacheable) in enumerate(cases, start=1):
        session_id = f"bench-500-{idx:03d}"

        payload, latency_ms = post_chat(session_id, text)
        hit = cache_hit(payload)
        ctype = cache_type(payload)

        gen_completed = generator_completed(payload)
        if before_generator_completed is None:
            before_generator_completed = gen_completed

        if expected_cacheable:
            expected_cacheable_total += 1
            if hit:
                expected_cacheable_hit += 1

        if expected_intent == "order":
            order_total += 1
            if hit:
                order_false_hit += 1

        if hit:
            total_hits += 1
            hit_latencies.append(latency_ms)
            # Nếu cache hit đúng, generator.completed không nên tăng cho chính request đó.
            # Ở đây ta log để audit bằng rows.
            generator_bypass_hits += 1
        else:
            miss_latencies.append(latency_ms)

        all_latencies.append(latency_ms)

        row = {
            "idx": idx,
            "text": text,
            "expected_intent_group": expected_intent,
            "actual_intent": payload.get("intent"),
            "expected_cacheable": expected_cacheable,
            "cache_hit": hit,
            "cache_type": ctype,
            "latency_ms": round(latency_ms, 2),
            "cache_lookup_text": cache_lookup_text(payload),
            "extractor_backend": extraction_backend(payload),
            "generator_completed": gen_completed,
            "answer": payload.get("answer", ""),
        }
        rows.append(row)

        if idx <= 20 or idx % 50 == 0:
            print(
                f"{idx:03d} | {latency_ms:8.2f} ms | "
                f"expected={expected_intent:<10} actual={payload.get('intent'):<10} "
                f"hit={str(hit):<5} type={ctype:<8} | {text}"
            )

    cacheable_hit_rate = expected_cacheable_hit / max(1, expected_cacheable_total)
    overall_hit_rate = total_hits / max(1, len(cases))
    order_false_hit_rate = order_false_hit / max(1, order_total)

    summary = {
        "total_queries": len(cases),
        "seed_count": len(SEED_CASES),
        "overall_hit_rate": round(overall_hit_rate, 4),
        "expected_cacheable_total": expected_cacheable_total,
        "expected_cacheable_hit": expected_cacheable_hit,
        "expected_cacheable_hit_rate": round(cacheable_hit_rate, 4),
        "order_total": order_total,
        "order_false_hit": order_false_hit,
        "order_false_hit_rate": round(order_false_hit_rate, 4),
        "latency_all": summarize_latencies(all_latencies),
        "latency_cache_hit": summarize_latencies(hit_latencies),
        "latency_cache_miss": summarize_latencies(miss_latencies),
        "generator_bypass_hits": generator_bypass_hits,
        "targets": {
            "expected_cacheable_hit_rate": ">= 0.60",
            "order_false_hit_rate": "== 0.0",
            "cache_hit_latency_target": "<= 100-250ms realistic local target",
        },
        "pass": {
            "hit_rate": cacheable_hit_rate >= 0.60,
            "order_no_cache": order_false_hit_rate == 0.0,
            "cache_hit_latency_p95_under_250ms": summarize_latencies(hit_latencies)["p95_ms"] <= 250,
        },
    }

    report = {
        "summary": summary,
        "seed_results": seed_results,
        "rows": rows,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n[DONE] Saved {REPORT_PATH}")


if __name__ == "__main__":
    main()