import asyncio

from app.core.constants import Intent
from app.core.schemas import RAGQuery
from app.rag.retriever import graph_retriever


async def main():
    queries = [
        RAGQuery(query="Cho anh một ly bạc xỉu đá", intent=Intent.ORDER),
        RAGQuery(query="Wifi quán là gì?", intent=Intent.FAQ),
        RAGQuery(query="Có món nào ngon ít ngọt không?", intent=Intent.CONSULTANT),
    ]

    for q in queries:
        result = await graph_retriever.retrieve_by_vector(q)

        print("=" * 80)
        print("QUERY:", q.query)
        print("INTENT:", q.intent)
        print("SOURCES:", len(result.sources))

        for src in result.sources:
            print(f"- {src.source_type.value} | score={src.score:.4f} | {src.source_id}")
            print(src.text[:200])


if __name__ == "__main__":
    asyncio.run(main())