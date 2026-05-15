import asyncio

from app.core.constants import Intent
from app.core.schemas import RAGQuery
from app.rag.retriever import graph_retriever


TEST_CASES = [
    {
        "query": "Wifi quán là gì?",
        "intent": Intent.FAQ,
        "expected_keywords": ["wifi", "highlands_guest"],
    },
    {
        "query": "internet ở đây dùng sao?",
        "intent": Intent.FAQ,
        "expected_keywords": ["wifi", "highlands_guest"],
    },
    {
        "query": "mật khẩu mạng là gì em?",
        "intent": Intent.FAQ,
        "expected_keywords": ["wifi", "highlands_guest"],
    },
    {
        "query": "quán mấy giờ đóng cửa?",
        "intent": Intent.FAQ,
        "expected_keywords": ["22:00", "mở cửa"],
    },
    {
        "query": "có giao hàng không?",
        "intent": Intent.FAQ,
        "expected_keywords": ["giao hàng", "đối tác"],
    },
    {
        "query": "Cho anh một ly bạc xỉu đá",
        "intent": Intent.ORDER,
        "expected_keywords": ["bạc xỉu"],
    },
    {
        "query": "Anh muốn gọi cappuccino ít ngọt",
        "intent": Intent.ORDER,
        "expected_keywords": ["cappuccino", "ít ngọt"],
    },
    {
        "query": "Cho tôi một phần bánh mì que",
        "intent": Intent.ORDER,
        "expected_keywords": ["bánh mì"],
    },
    {
        "query": "Có trà sen không?",
        "intent": Intent.ORDER,
        "expected_keywords": ["trà sen", "tea"],
    },
    {
        "query": "Tôi muốn uống cà phê đen đá",
        "intent": Intent.ORDER,
        "expected_keywords": ["cà phê đen"],
    },
    {
        "query": "Có món nào ít ngọt không?",
        "intent": Intent.CONSULTANT,
        "expected_keywords": ["ít ngọt", "giảm đường"],
    },
    {
        "query": "Uống gì ngon mà không quá ngọt?",
        "intent": Intent.CONSULTANT,
        "expected_keywords": ["ít ngọt", "trà", "cà phê"],
    },
    {
        "query": "Có món nào rẻ không?",
        "intent": Intent.CONSULTANT,
        "expected_keywords": ["rẻ", "giá"],
    },
    {
        "query": "Gợi ý món cà phê dễ uống",
        "intent": Intent.CONSULTANT,
        "expected_keywords": ["coffee", "cà phê"],
    },
    {
        "query": "Có đồ uống đá xay nào ngon không?",
        "intent": Intent.CONSULTANT,
        "expected_keywords": ["freeze", "đá xay"],
    },
    {
        "query": "Quán có những size nào?",
        "intent": Intent.FAQ,
        "expected_keywords": ["size", "s", "m", "l"],
    },
    {
        "query": "Có thanh toán momo không?",
        "intent": Intent.FAQ,
        "expected_keywords": ["thanh toán", "momo", "payment"],
    },
    {
        "query": "Cho em latte size M",
        "intent": Intent.ORDER,
        "expected_keywords": ["latte"],
    },
    {
        "query": "Món nào hợp cho người thích trà?",
        "intent": Intent.CONSULTANT,
        "expected_keywords": ["trà", "tea"],
    },
    {
        "query": "Có bánh ngọt ăn kèm không?",
        "intent": Intent.CONSULTANT,
        "expected_keywords": ["bánh", "food"],
    },
]


def normalize(text: str) -> str:
    return text.lower()


def is_hit(text: str, expected_keywords: list[str]) -> bool:
    normalized = normalize(text)
    return any(keyword.lower() in normalized for keyword in expected_keywords)


async def main():
    total = len(TEST_CASES)
    top1_hits = 0
    top5_hits = 0

    print("=" * 80)
    print("RAG RETRIEVAL MINI BENCHMARK")
    print("=" * 80)

    for idx, case in enumerate(TEST_CASES, start=1):
        result = await graph_retriever.retrieve_auto(
            RAGQuery(
                query=case["query"],
                intent=case["intent"],
                top_k=5,
            )
        )

        sources = result.sources
        top1_text = sources[0].text if sources else ""
        top5_text = "\n".join(source.text for source in sources)

        top1_hit = is_hit(top1_text, case["expected_keywords"])
        top5_hit = is_hit(top5_text, case["expected_keywords"])

        top1_hits += int(top1_hit)
        top5_hits += int(top5_hit)

        print(f"\n[{idx}/{total}] {case['query']}")
        print(f"Intent: {case['intent'].value}")
        print(f"Mode: {result.metadata.get('retrieval_mode')}")
        print(f"Top-1 hit: {top1_hit}")
        print(f"Top-5 hit: {top5_hit}")

        if sources:
            print(
                f"Top source: "
                f"{sources[0].source_type.value} | "
                f"{sources[0].score:.4f} | "
                f"{sources[0].source_id}"
            )
            print(sources[0].text[:180])

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total: {total}")
    print(f"Top-1 accuracy: {top1_hits / total:.2%}")
    print(f"Top-5 accuracy: {top5_hits / total:.2%}")


if __name__ == "__main__":
    asyncio.run(main())