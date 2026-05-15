import csv
import json
from pathlib import Path

from app.rag.neo4j_client import neo4j_client


DATA_DIR = Path("data/mock")

MENU_PATH = DATA_DIR / "menu.csv"
FAQ_PATH = DATA_DIR / "faq.csv"
DOC_PATH = DATA_DIR / "docs.jsonl"

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


def create_constraints() -> None:
    queries = [
        """
        CREATE CONSTRAINT menu_id_unique IF NOT EXISTS
        FOR (m:MenuItem)
        REQUIRE m.id IS UNIQUE
        """,
        """
        CREATE CONSTRAINT faq_id_unique IF NOT EXISTS
        FOR (f:FAQ)
        REQUIRE f.id IS UNIQUE
        """,
        """
        CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
        FOR (c:Chunk)
        REQUIRE c.id IS UNIQUE
        """,
        """
        CREATE CONSTRAINT entity_key_unique IF NOT EXISTS
        FOR (e:Entity)
        REQUIRE e.key IS UNIQUE
        """,
        """
        CREATE CONSTRAINT category_name_unique IF NOT EXISTS
        FOR (c:Category)
        REQUIRE c.name IS UNIQUE
        """,
    ]

    for q in queries:
        neo4j_client.execute_query(q)

    print("Constraints ready")


def clear_graph() -> None:
    neo4j_client.execute_query(
        """
        MATCH (n)
        DETACH DELETE n
        """
    )
    print("Cleared graph")


def ingest_menu() -> None:
    count = 0

    with MENU_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Create MenuItem and Category
            neo4j_client.execute_query(
                """
                MERGE (m:MenuItem {id: $id})
                SET
                    m.name_vi = $name_vi,
                    m.name_en = $name_en,
                    m.price = $price,
                    m.size = $size,
                    m.category = $category,
                    m.description = $description,
                    m.available = $available

                MERGE (c:Category {name: $category})
                MERGE (m)-[:BELONGS_TO]->(c)
                """,
                {
                    "id": row["id"],
                    "name_vi": row["name_vi"],
                    "name_en": row["name_en"],
                    "price": int(row["price"]),
                    "size": row["size"],
                    "category": row["category"],
                    "description": row["description"],
                    "available": row["available"].strip().lower() == "true",
                },
            )

            # Ingredients
            ingredients = [
                x.strip()
                for x in row["ingredients"].split("|")
                if x.strip()
            ]

            for ingredient in ingredients:
                neo4j_client.execute_query(
                    """
                    MERGE (e:Entity {key: $key})
                    SET
                        e.name = $name,
                        e.type = 'ingredient'
                    WITH e
                    MATCH (m:MenuItem {id: $menu_id})
                    MERGE (m)-[:HAS_INGREDIENT]->(e)
                    """,
                    {
                        "key": ingredient.lower(),
                        "name": ingredient,
                        "menu_id": row["id"],
                    },
                )

            # Tags
            tags = [
                x.strip()
                for x in row["tags"].split("|")
                if x.strip()
            ]

            for tag in tags:
                neo4j_client.execute_query(
                    """
                    MERGE (e:Entity {key: $key})
                    SET
                        e.name = $name,
                        e.type = 'tag'
                    WITH e
                    MATCH (m:MenuItem {id: $menu_id})
                    MERGE (m)-[:HAS_TAG]->(e)
                    """,
                    {
                        "key": f"tag::{tag.lower()}",
                        "name": tag,
                        "menu_id": row["id"],
                    },
                )

            count += 1

    print(f"MenuItems: {count}")


def ingest_faq() -> None:
    count = 0

    with FAQ_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            neo4j_client.execute_query(
                """
                MERGE (f:FAQ {id: $id})
                SET
                    f.topic = $topic,
                    f.question = $question,
                    f.answer = $answer,
                    f.language = $language,
                    f.source_file = $source_file
                """,
                row,
            )
            count += 1

    print(f"FAQ: {count}")


def ingest_docs() -> None:
    count = 0

    with DOC_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)

            # Create Chunk
            neo4j_client.execute_query(
                """
                MERGE (c:Chunk {id: $id})
                SET
                    c.text = $text,
                    c.language = $language,
                    c.chunk_index = $chunk_index,
                    c.source_file = $source_file
                """,
                doc,
            )

            # Create Entities and relationships
            for entity in doc.get("entities", []):
                neo4j_client.execute_query(
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
                        "chunk_id": doc["id"],
                    },
                )

            count += 1

    print(f"Chunks: {count}")

def _contains_any(text: str, keywords: list[str]) -> bool:
    text = (text or "").lower()
    return any(keyword.lower() in text for keyword in keywords)


def ingest_domain_entities() -> None:
    for entity in DOMAIN_ENTITIES:
        neo4j_client.execute_query(
            """
            MERGE (e:Entity {key: $key})
            SET
                e.name = $name,
                e.type = $type,
                e.keywords = $keywords
            """,
            entity,
        )

    print(f"Domain entities: {len(DOMAIN_ENTITIES)}")


def link_faq_domain_mentions() -> None:
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

    linked = 0

    for row in rows:
        text = " ".join(
            [
                str(row.get("topic") or ""),
                str(row.get("question") or ""),
                str(row.get("answer") or ""),
            ]
        ).lower()

        for entity in DOMAIN_ENTITIES:
            if _contains_any(text, entity["keywords"]):
                neo4j_client.execute_query(
                    """
                    MATCH (f:FAQ {id: $faq_id})
                    MATCH (e:Entity {key: $entity_key})
                    MERGE (f)-[:MENTIONS]->(e)
                    """,
                    {
                        "faq_id": row["id"],
                        "entity_key": entity["key"],
                    },
                )
                linked += 1

    print(f"FAQ -> MENTIONS -> Entity: {linked}")


def link_chunk_domain_mentions() -> None:
    rows = neo4j_client.execute_query(
        """
        MATCH (c:Chunk)
        RETURN
            c.id AS id,
            c.text AS text
        """
    )

    linked = 0

    for row in rows:
        text = str(row.get("text") or "").lower()

        for entity in DOMAIN_ENTITIES:
            if _contains_any(text, entity["keywords"]):
                neo4j_client.execute_query(
                    """
                    MATCH (c:Chunk {id: $chunk_id})
                    MATCH (e:Entity {key: $entity_key})
                    MERGE (c)-[:MENTIONS]->(e)
                    """,
                    {
                        "chunk_id": row["id"],
                        "entity_key": entity["key"],
                    },
                )
                linked += 1

    print(f"Chunk -> MENTIONS -> Entity: {linked}")


def link_menu_domain_mentions() -> None:
    rows = neo4j_client.execute_query(
        """
        MATCH (m:MenuItem)
        RETURN
            m.id AS id,
            m.name_vi AS name_vi,
            m.name_en AS name_en,
            m.description AS description,
            m.category AS category
        """
    )

    linked = 0

    for row in rows:
        text = " ".join(
            [
                str(row.get("name_vi") or ""),
                str(row.get("name_en") or ""),
                str(row.get("description") or ""),
                str(row.get("category") or ""),
            ]
        ).lower()

        for entity in DOMAIN_ENTITIES:
            if _contains_any(text, entity["keywords"]):
                neo4j_client.execute_query(
                    """
                    MATCH (m:MenuItem {id: $menu_id})
                    MATCH (e:Entity {key: $entity_key})
                    MERGE (m)-[:MENTIONS]->(e)
                    """,
                    {
                        "menu_id": row["id"],
                        "entity_key": entity["key"],
                    },
                )
                linked += 1

    print(f"MenuItem -> MENTIONS -> Entity: {linked}")


def build_domain_mentions() -> None:
    ingest_domain_entities()
    link_faq_domain_mentions()
    link_chunk_domain_mentions()
    link_menu_domain_mentions()

def graph_stats() -> None:
    queries = {
        "MenuItems": "MATCH (n:MenuItem) RETURN count(n) AS count",
        "FAQ": "MATCH (n:FAQ) RETURN count(n) AS count",
        "Chunk": "MATCH (n:Chunk) RETURN count(n) AS count",
        "Entity": "MATCH (n:Entity) RETURN count(n) AS count",
        "Category": "MATCH (n:Category) RETURN count(n) AS count",
        "Relationships": "MATCH ()-[r]->() RETURN count(r) AS count",
    }

    print("\n=== GRAPH STATS ===")

    for name, query in queries.items():
        result = neo4j_client.execute_query(query)
        print(f"{name}: {result[0]['count']}")


def build_chunk_next_relationships() -> None:
    rows = neo4j_client.execute_query(
        """
        MATCH (c:Chunk)
        RETURN c.id AS id
        ORDER BY c.id
        """
    )

    for i in range(len(rows) - 1):
        current_id = rows[i]["id"]
        next_id = rows[i + 1]["id"]

        neo4j_client.execute_query(
            """
            MATCH (c1:Chunk {id: $current_id})
            MATCH (c2:Chunk {id: $next_id})
            MERGE (c1)-[:NEXT]->(c2)
            """,
            {
                "current_id": current_id,
                "next_id": next_id,
            },
        )

    print("Chunk NEXT relationships built.")

def main() -> None:
    assert neo4j_client.verify_connection(), "Neo4j connection failed"

    clear_graph()
    create_constraints()

    ingest_menu()
    ingest_faq()
    ingest_docs()
    build_chunk_next_relationships()
    build_domain_mentions()

    graph_stats()

    print("\nNeo4j ingestion completed")


if __name__ == "__main__":
    main()