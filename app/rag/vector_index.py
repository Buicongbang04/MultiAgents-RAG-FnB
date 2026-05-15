from app.core.config import get_settings
from app.core.logging import get_logger
from app.rag.embedding_client import get_embedding_client
from app.rag.neo4j_client import neo4j_client

logger = get_logger(__name__)


async def embed_menu_items() -> int:
    client = get_embedding_client()

    rows = neo4j_client.execute_query(
        """
        MATCH (m:MenuItem)
        RETURN
            m.id AS id,
            m.name_vi AS name_vi,
            m.name_en AS name_en,
            m.category AS category,
            m.description AS description,
            m.price AS price,
            m.size AS size
        """
    )

    updated = 0

    for row in rows:
        text = (
            f"{row.get('name_vi') or ''} | "
            f"{row.get('name_en') or ''} | "
            f"{row.get('category') or ''} | "
            f"{row.get('description') or ''} | "
            f"{row.get('price') or ''} VND | "
            f"Size {row.get('size') or ''}"
        )

        embedding = await client.embed_text(text)

        neo4j_client.execute_query(
            """
            MATCH (m:MenuItem {id: $id})
            SET
                m.embedding = $embedding,
                m.embedding_model = $embedding_model
            """,
            {
                "id": row["id"],
                "embedding": embedding,
                "embedding_model": get_settings().embedding_model,
            },
        )

        updated += 1

    logger.info("Embedded MenuItem nodes: %d", updated)
    return updated


async def embed_faq_items() -> int:
    client = get_embedding_client()

    rows = neo4j_client.execute_query(
        """
        MATCH (f:FAQ)
        RETURN
            f.id AS id,
            f.topic AS topic,
            f.question AS question,
            f.answer AS answer
        """
    )

    updated = 0

    for row in rows:
        text = (
            f"Topic: {row.get('topic') or ''}\n"
            f"Question: {row.get('question') or ''}\n"
            f"Answer: {row.get('answer') or ''}"
        )

        embedding = await client.embed_text(text)

        neo4j_client.execute_query(
            """
            MATCH (f:FAQ {id: $id})
            SET
                f.embedding = $embedding,
                f.embedding_model = $embedding_model
            """,
            {
                "id": row["id"],
                "embedding": embedding,
                "embedding_model": get_settings().embedding_model,
            },
        )

        updated += 1

    logger.info("Embedded FAQ nodes: %d", updated)
    return updated


async def embed_chunk_items() -> int:
    client = get_embedding_client()

    rows = neo4j_client.execute_query(
        """
        MATCH (c:Chunk)
        RETURN c.id AS id, c.text AS text
        """
    )

    updated = 0

    for row in rows:
        embedding = await client.embed_text(row.get("text") or "")

        neo4j_client.execute_query(
            """
            MATCH (c:Chunk {id: $id})
            SET
                c.embedding = $embedding,
                c.embedding_model = $embedding_model
            """,
            {
                "id": row["id"],
                "embedding": embedding,
                "embedding_model": get_settings().embedding_model,
            },
        )

        updated += 1

    logger.info("Embedded Chunk nodes: %d", updated)
    return updated


async def embed_all_graph_nodes() -> dict:
    menu_count = await embed_menu_items()
    faq_count = await embed_faq_items()
    chunk_count = await embed_chunk_items()

    return {
        "menu_items": menu_count,
        "faq_items": faq_count,
        "chunks": chunk_count,
    }