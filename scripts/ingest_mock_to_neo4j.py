import csv
import json
from pathlib import Path

from app.rag.neo4j_client import neo4j_client


DATA_DIR = Path("data/mock")

MENU_PATH = DATA_DIR / "menu.csv"
FAQ_PATH = DATA_DIR / "faq.csv"
DOC_PATH = DATA_DIR / "docs.jsonl"


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
        CREATE CONSTRAINT entity_name_unique IF NOT EXISTS
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
    neo4j_client.execute_query("""
    MATCH (n)
    DETACH DELETE n
    """)
    print("Cleared graph")


def ingest_menu() -> None:
    count = 0

    with MENU_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            query = """
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
            """

            neo4j_client.execute_query(
                query,
                {
                    "id": row["id"],
                    "name_vi": row["name_vi"],
                    "name_en": row["name_en"],
                    "price": int(row["price"]),
                    "size": row["size"],
                    "category": row["category"],
                    "description": row["description"],
                    "available": row["available"] == "true",
                },
            )

            ingredients = row["ingredients"].split("|")
            tags = row["tags"].split("|")

            for ingredient in ingredients:
                neo4j_client.execute_query(
                    """
                    MERGE (e:Entity {key: $key})
                    SET
                        e.name = $name,
                        e.type = 'ingredient'

                    MATCH (m:MenuItem {id: $menu_id})
                    MERGE (m)-[:HAS_INGREDIENT]->(e)
                    """,
                    {
                        "key": ingredient.lower(),
                        "name": ingredient,
                        "menu_id": row["id"],
                    },
                )

            for tag in tags:
                neo4j_client.execute_query(
                    """
                    MERGE (e:Entity {key: $key})
                    SET
                        e.name = $name,
                        e.type = 'tag'

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

            for entity in doc["entities"]:
                neo4j_client.execute_query(
                    """
                    MERGE (e:Entity {key: $key})
                    SET
                        e.name = $name,
                        e.type = $type

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


def graph_stats() -> None:
    queries = {
        "MenuItems": "MATCH (n:MenuItem) RETURN count(n) as count",
        "FAQ": "MATCH (n:FAQ) RETURN count(n) as count",
        "Chunk": "MATCH (n:Chunk) RETURN count(n) as count",
        "Entity": "MATCH (n:Entity) RETURN count(n) as count",
        "Category": "MATCH (n:Category) RETURN count(n) as count",
        "Relationships": "MATCH ()-[r]->() RETURN count(r) as count",
    }

    print("\n=== GRAPH STATS ===")

    for name, query in queries.items():
        result = neo4j_client.execute_query(query)
        print(f"{name}: {result[0]['count']}")


def main() -> None:
    assert neo4j_client.verify_connection(), "Neo4j connection failed"

    clear_graph()
    create_constraints()

    ingest_menu()
    ingest_faq()
    ingest_docs()

    graph_stats()

    print("\nNeo4j ingestion completed")


if __name__ == "__main__":
    main()