import asyncio

from app.core.constants import Intent
from app.core.schemas import RAGQuery
from app.rag.retriever import graph_retriever


async def main():

    queries = [
        ("Cho anh một ly bạc xỉu đá", Intent.ORDER),
        ("internet quán là gì", Intent.FAQ),
        ("uống gì ít ngọt ngon", Intent.CONSULTANT),
    ]

    for text, intent in queries:

        result = await graph_retriever.retrieve_hybrid_with_graph_expansion(
            RAGQuery(
                query=text,
                intent=intent,
                top_k=8,
            ),
            keyword_weight=0.45 if intent == Intent.FAQ else 0.65,
            vector_weight=0.55 if intent == Intent.FAQ else 0.35,
        )

        print("=" * 80)
        print("QUERY:", text)
        print("MODE:", result.metadata)

        for src in result.sources:
            print(
                f"[{src.source_type.value}] "
                f"{src.score:.4f} "
                f"{src.source_id} "
                f"{src.metadata}"
            )
            print(src.text[:220])


if __name__ == "__main__":
    asyncio.run(main())