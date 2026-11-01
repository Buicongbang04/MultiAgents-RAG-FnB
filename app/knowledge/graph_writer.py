"""Project Postgres catalog DTOs into the Neo4j graph (Postgres is the source of truth).

Cypher semantics are unchanged from the former scripts/ingest_mock_to_neo4j.py: the retriever
(app/rag/retriever.py) and app/rag/menu.py depend on these labels/properties/relationships.
"""

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.catalog.repository import list_faqs, list_policy_chunks, list_products
from app.catalog.schemas import FaqDTO, PolicyChunkDTO, ProductDTO
from app.core.logging import get_logger
from app.knowledge.indexes import ensure_constraints
from app.rag.neo4j_client import Neo4jClient

logger = get_logger(__name__)

# FAQ rows no longer carry source_file in Postgres; keep the legacy graph value.
FAQ_SOURCE_FILE = "highlands_faq.csv"

DOMAIN_ENTITIES = [
    {
        "key": "topic::wifi",
        "name": "wifi",
        "type": "faq_topic",
        "keywords": ["wifi", "wi-fi", "internet", "mạng", "mật khẩu", "password"],
    },
    {
        "key": "topic::opening_hours",
        "name": "opening_hours",
        "type": "faq_topic",
        "keywords": ["giờ", "mở cửa", "đóng cửa", "open", "close", "opening_hours"],
    },
    {
        "key": "topic::delivery",
        "name": "delivery",
        "type": "faq_topic",
        "keywords": ["delivery", "giao hàng", "ship", "mang đi", "take away"],
    },
    {
        "key": "topic::size",
        "name": "size",
        "type": "faq_topic",
        "keywords": ["size", "kích cỡ", "cỡ ly"],
    },
    {
        "key": "topic::payment",
        "name": "payment",
        "type": "faq_topic",
        "keywords": ["thanh toán", "payment", "tiền mặt", "chuyển khoản", "momo", "visa"],
    },
    {
        "key": "preference::low_sugar",
        "name": "ít ngọt",
        "type": "preference",
        "keywords": ["ít ngọt", "không ngọt", "giảm đường", "less sugar", "low sugar"],
    },
    {
        "key": "preference::budget",
        "name": "giá rẻ",
        "type": "preference",
        "keywords": ["giá rẻ", "rẻ", "tiết kiệm", "budget", "cheap"],
    },
    {
        "key": "category::coffee",
        "name": "coffee",
        "type": "category",
        "keywords": ["coffee", "cà phê", "cafe", "bạc xỉu", "latte", "cappuccino", "espresso"],
    },
    {
        "key": "category::tea",
        "name": "tea",
        "type": "category",
        "keywords": ["tea", "trà", "trà sen", "trà đào", "trà xanh"],
    },
    {
        "key": "category::freeze",
        "name": "freeze",
        "type": "category",
        "keywords": ["freeze", "đá xay", "chocolate freeze", "caramel freeze"],
    },
    {
        "key": "category::food",
        "name": "food",
        "type": "category",
        "keywords": ["food", "bánh", "bánh mì", "bánh ngọt", "đồ ăn"],
    },
]


@dataclass(frozen=True)
class ProjectionStats:
    products: int
    menu_items: int
    faqs: int
    chunks: int
    entities: int
    categories: int
    relationships: int


def _vnd(amount: int) -> str:
    """35000 -> '35.000đ' (Vietnamese thousands separator)."""
    return f"{amount:,}".replace(",", ".") + "đ"


def variant_description(product: ProductDTO, size: str, price_vnd: int) -> str:
    """Per-variant description equivalent to the legacy CSV text ("<name> size S (35.000đ) — ...").

    Product.description is the CSV text with the size clause removed, so it normally starts with
    the name; the size clause is re-inserted right after it.
    """
    clause = f"size {size} ({_vnd(price_vnd)})"
    if product.description.startswith(product.name_vi):
        rest = product.description[len(product.name_vi):]
        return f"{product.name_vi} {clause}{rest}"
    return f"{product.name_vi} {clause} — {product.description}"


def clear_graph(graph: Neo4jClient) -> None:
    graph.execute_query("MATCH (n) DETACH DELETE n")


def write_menu(graph: Neo4jClient, products: list[ProductDTO]) -> int:
    count = 0
    for product in products:
        available = product.status == "active"
        for variant in product.variants:
            graph.execute_query(
                """
                MERGE (m:MenuItem {id: $id})
                SET
                    m.product_id = $product_id,
                    m.name_vi = $name_vi,
                    m.name_en = $name_en,
                    m.price = $price,
                    m.size = $size,
                    m.category = $category,
                    m.ingredients = $ingredients,
                    m.tags = $tags,
                    m.description = $description,
                    m.available = $available

                MERGE (c:Category {name: $category})
                MERGE (m)-[:BELONGS_TO]->(c)
                """,
                {
                    "id": variant.id,
                    "product_id": product.id,
                    "name_vi": product.name_vi,
                    "name_en": product.name_en,
                    "price": variant.price_vnd,
                    "size": variant.size,
                    "category": product.category,
                    "ingredients": product.ingredients,
                    "tags": product.tags,
                    "description": variant_description(product, variant.size, variant.price_vnd),
                    "available": available,
                },
            )

            for ingredient in product.ingredients:
                graph.execute_query(
                    """
                    MERGE (e:Entity {key: $key})
                    SET
                        e.name = $name,
                        e.type = 'ingredient'
                    WITH e
                    MATCH (m:MenuItem {id: $menu_id})
                    MERGE (m)-[:HAS_INGREDIENT]->(e)
                    """,
                    {"key": ingredient.lower(), "name": ingredient, "menu_id": variant.id},
                )

            for tag in product.tags:
                graph.execute_query(
                    """
                    MERGE (e:Entity {key: $key})
                    SET
                        e.name = $name,
                        e.type = 'tag'
                    WITH e
                    MATCH (m:MenuItem {id: $menu_id})
                    MERGE (m)-[:HAS_TAG]->(e)
                    """,
                    {"key": f"tag::{tag.lower()}", "name": tag, "menu_id": variant.id},
                )
            count += 1
    return count


def write_faqs(graph: Neo4jClient, faqs: list[FaqDTO]) -> int:
    for faq in faqs:
        graph.execute_query(
            """
            MERGE (f:FAQ {id: $id})
            SET
                f.topic = $topic,
                f.question = $question,
                f.answer = $answer,
                f.language = $language,
                f.source_file = $source_file
            """,
            {**faq.model_dump(), "source_file": FAQ_SOURCE_FILE},
        )
    return len(faqs)


def write_chunks(graph: Neo4jClient, chunks: list[PolicyChunkDTO]) -> int:
    for chunk in chunks:
        graph.execute_query(
            """
            MERGE (c:Chunk {id: $id})
            SET
                c.text = $text,
                c.language = $language,
                c.chunk_index = $chunk_index,
                c.source_file = $source_file
            """,
            chunk.model_dump(exclude={"entities"}),
        )
        for entity in chunk.entities:
            graph.execute_query(
                """
                MERGE (e:Entity {key: $key})
                SET
                    e.name = $name,
                    e.type = $type
                WITH e
                MATCH (c:Chunk {id: $chunk_id})
                MERGE (c)-[:MENTIONS]->(e)
                """,
                {
                    "key": entity["normalized_name"],
                    "name": entity["name"],
                    "type": entity["type"],
                    "chunk_id": chunk.id,
                },
            )
    return len(chunks)


def _contains_any(text: str, keywords: list[str]) -> bool:
    text = (text or "").lower()
    return any(keyword.lower() in text for keyword in keywords)


def build_chunk_next_relationships(graph: Neo4jClient) -> None:
    rows = graph.execute_query("MATCH (c:Chunk) RETURN c.id AS id ORDER BY c.id")
    for current, nxt in zip(rows, rows[1:]):
        graph.execute_query(
            """
            MATCH (c1:Chunk {id: $current_id})
            MATCH (c2:Chunk {id: $next_id})
            MERGE (c1)-[:NEXT]->(c2)
            """,
            {"current_id": current["id"], "next_id": nxt["id"]},
        )


def ingest_domain_entities(graph: Neo4jClient) -> None:
    for entity in DOMAIN_ENTITIES:
        graph.execute_query(
            """
            MERGE (e:Entity {key: $key})
            SET
                e.name = $name,
                e.type = $type,
                e.keywords = $keywords
            """,
            entity,
        )


def _link_mentions(graph: Neo4jClient, label: str, id_key: str, query: str, fields: list[str]) -> int:
    rows = graph.execute_query(query)
    linked = 0
    for row in rows:
        text = " ".join(str(row.get(f) or "") for f in fields).lower()
        for entity in DOMAIN_ENTITIES:
            if _contains_any(text, entity["keywords"]):
                graph.execute_query(
                    f"""
                    MATCH (n:{label} {{id: $id}})
                    MATCH (e:Entity {{key: $entity_key}})
                    MERGE (n)-[:MENTIONS]->(e)
                    """,
                    {"id": row[id_key], "entity_key": entity["key"]},
                )
                linked += 1
    return linked


def build_domain_mentions(graph: Neo4jClient) -> None:
    ingest_domain_entities(graph)
    _link_mentions(
        graph,
        "FAQ",
        "id",
        "MATCH (f:FAQ) RETURN f.id AS id, f.topic AS topic, f.question AS question, "
        "f.answer AS answer",
        ["topic", "question", "answer"],
    )
    _link_mentions(
        graph,
        "Chunk",
        "id",
        "MATCH (c:Chunk) RETURN c.id AS id, c.text AS text",
        ["text"],
    )
    _link_mentions(
        graph,
        "MenuItem",
        "id",
        "MATCH (m:MenuItem) RETURN m.id AS id, m.name_vi AS name_vi, m.name_en AS name_en, "
        "m.description AS description, m.category AS category",
        ["name_vi", "name_en", "description", "category"],
    )


def graph_stats(graph: Neo4jClient) -> dict[str, int]:
    queries = {
        "menu_items": "MATCH (n:MenuItem) RETURN count(n) AS c",
        "faqs": "MATCH (n:FAQ) RETURN count(n) AS c",
        "chunks": "MATCH (n:Chunk) RETURN count(n) AS c",
        "entities": "MATCH (n:Entity) RETURN count(n) AS c",
        "categories": "MATCH (n:Category) RETURN count(n) AS c",
        "relationships": "MATCH ()-[r]->() RETURN count(r) AS c",
    }
    return {k: graph.execute_query(q)[0]["c"] for k, q in queries.items()}


async def project_catalog(db: AsyncSession, graph: Neo4jClient) -> ProjectionStats:
    """Clear the graph and rebuild it from Postgres. Sold-out/hidden products stay as
    unavailable MenuItems (retrieval filters on coalesce(m.available, true))."""
    products = await list_products(db, include_hidden=True)
    faqs = await list_faqs(db)
    chunks = await list_policy_chunks(db)

    clear_graph(graph)
    ensure_constraints(graph)
    write_menu(graph, products)
    write_faqs(graph, faqs)
    write_chunks(graph, chunks)
    build_chunk_next_relationships(graph)
    build_domain_mentions(graph)

    stats = graph_stats(graph)
    logger.info("Projected catalog into Neo4j: %s", stats)
    return ProjectionStats(products=len(products), **stats)
