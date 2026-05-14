import asyncio

from app.core.constants import Intent
from app.core.schemas import RAGQuery
from app.rag.retriever import graph_retriever


async def main():

    samples = [
        ("bạc xỉu", Intent.ORDER),
        ("món ngon ít ngọt", Intent.CONSULTANT),
        ("wifi", Intent.FAQ),
        ("mấy giờ mở cửa", Intent.FAQ),
    ]

    for query, intent in samples:
        print("\n" + "=" * 80)
        print(query, intent.value)

        result = await graph_retriever.retrieve(
            RAGQuery(
                query=query,
                intent=intent,
            )
        )

        print("num sources:", len(result.sources))

        for s in result.sources[:3]:
            print(
                s.source_type,
                s.score,
                s.text[:120],
            )

        print("\nCONTEXT:")
        print(result.context_text[:500])


asyncio.run(main())