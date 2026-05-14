from collections import defaultdict

from app.core.config import get_settings
from app.core.constants import Intent, SourceType
from app.core.logging import get_logger
from app.core.schemas import (
    RAGQuery,
    RAGResult,
    RetrievedSource,
)
from app.rag.neo4j_client import neo4j_client

logger = get_logger(__name__)


class GraphRetriever:
    """
    MVP Graph Retrieval.

    Có thể: 
    - keyword retrieval
    - graph expansion đơn giản
    - merge/rank rule-based

    Chưa dùng:
    - embeddings
    - vector db
    - reranker
    - hybrid retrieval
    """

    def __init__(self) -> None:
        self.settings = get_settings()

    async def retrieve(
        self,
        rag_query: RAGQuery,
    ) -> RAGResult:

        intent = rag_query.intent
        query = rag_query.query.strip()

        if intent == Intent.ORDER:
            sources = self._retrieve_menu(query)

        elif intent == Intent.CONSULTANT:
            menu_sources = self._retrieve_menu(query)
            doc_sources = self._retrieve_docs(query)

            sources = self._merge_sources(
                menu_sources,
                doc_sources,
            )

        elif intent == Intent.FAQ:
            faq_sources = self._retrieve_faq(query)
            doc_sources = self._retrieve_docs(query)

            sources = self._merge_sources(
                faq_sources,
                doc_sources,
            )

        else:
            sources = []

        sources = self._deduplicate_sources(sources)
        sources = sorted(
            sources,
            key=lambda x: x.score,
            reverse=True,
        )

        top_sources = sources[: self.settings.rag_top_k]

        context_text = self._build_context(
            top_sources,
        )

        logger.info(
            "Retrieval query='%s' intent=%s sources=%d",
            query,
            intent.value if intent else None,
            len(top_sources),
        )

        return RAGResult(
            query=query,
            sources=top_sources,
            context_text=context_text,
        )

    def _retrieve_menu(
        self,
        query: str,
    ) -> list[RetrievedSource]:

        cypher = """
        MATCH (m:MenuItem)
        WHERE
            toLower(m.name_vi) CONTAINS toLower($query)
            OR toLower(m.description) CONTAINS toLower($query)
            OR EXISTS {
                MATCH (m)-[:HAS_TAG]->(e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($query)
            }

        RETURN
            m.id as id,
            m.name_vi as name,
            m.description as text,
            m.price as price,
            m.category as category,
            m.size as size
        LIMIT 20
        """

        rows = neo4j_client.execute_query(
            cypher,
            {"query": query},
        )

        sources = []

        for row in rows:
            text = (
                f"{row['name']} | "
                f"{row['category']} | "
                f"{row['price']} VND | "
                f"{row['text']}"
            )

            sources.append(
                RetrievedSource(
                    source_id=row["id"],
                    source_type=SourceType.MENU,
                    text=text,
                    score=0.90,
                    metadata={
                        "price": row["price"],
                        "category": row["category"],
                        "size": row["size"],
                    },
                )
            )

        return sources

    def _retrieve_faq(
        self,
        query: str,
    ) -> list[RetrievedSource]:

        cypher = """
        MATCH (f:FAQ)
        WHERE
            toLower(f.question) CONTAINS toLower($query)
            OR toLower(f.topic) CONTAINS toLower($query)
            OR toLower(f.answer) CONTAINS toLower($query)

        RETURN
            f.id as id,
            f.question as question,
            f.answer as answer
        LIMIT 20
        """

        rows = neo4j_client.execute_query(
            cypher,
            {"query": query},
        )

        return [
            RetrievedSource(
                source_id=row["id"],
                source_type=SourceType.FAQ,
                text=f"Q: {row['question']}\nA: {row['answer']}",
                score=0.88,
            )
            for row in rows
        ]

    def _retrieve_docs(
        self,
        query: str,
    ) -> list[RetrievedSource]:

        cypher = """
        MATCH (c:Chunk)

        OPTIONAL MATCH
        (c)-[:MENTIONS]->(e:Entity)

        WHERE
            toLower(c.text) CONTAINS toLower($query)
            OR toLower(e.name) CONTAINS toLower($query)

        RETURN DISTINCT
            c.id as id,
            c.text as text
        LIMIT 20
        """

        rows = neo4j_client.execute_query(
            cypher,
            {"query": query},
        )

        return [
            RetrievedSource(
                source_id=row["id"],
                source_type=SourceType.DOCUMENT,
                text=row["text"],
                score=0.75,
            )
            for row in rows
        ]

    def _merge_sources(
        self,
        *groups: list[RetrievedSource],
    ) -> list[RetrievedSource]:

        merged = []

        for group in groups:
            merged.extend(group)

        return merged

    def _deduplicate_sources(
        self,
        sources: list[RetrievedSource],
    ) -> list[RetrievedSource]:

        best_sources = {}

        for src in sources:
            existing = best_sources.get(src.source_id)

            if existing is None:
                best_sources[src.source_id] = src
                continue

            if src.score > existing.score:
                best_sources[src.source_id] = src

        return list(best_sources.values())

    def _build_context(
        self,
        sources: list[RetrievedSource],
    ) -> str:

        grouped = defaultdict(list)

        for src in sources:
            grouped[src.source_type.value].append(src.text)

        sections = []

        for source_type, values in grouped.items():
            block = "\n".join(values)
            sections.append(
                f"[{source_type.upper()}]\n{block}"
            )

        return "\n\n".join(sections)


graph_retriever = GraphRetriever()