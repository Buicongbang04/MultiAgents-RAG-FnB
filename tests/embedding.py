import asyncio

from app.rag.embedding_client import get_embedding_client


async def main():
    client = get_embedding_client()

    vec1 = await client.embed_text("bạc xỉu đá")
    vec2 = await client.embed_text("bạc xỉu đá")

    print("dim:", len(vec1))
    print("deterministic:", vec1 == vec2)
    print("non_zero:", sum(abs(x) for x in vec1) > 0)


if __name__ == "__main__":
    asyncio.run(main())