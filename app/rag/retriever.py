import re
from collections import defaultdict
import math

from app.core.config import get_settings
from app.core.constants import Intent, SourceType
from app.core.logging import get_logger
from app.core.schemas import RAGQuery, RAGResult, RetrievedSource
from app.rag.neo4j_client import neo4j_client
from app.rag.embedding_client import get_embedding_client

logger = get_logger(__name__)


STOPWORDS = {
    "cho", "anh", "chị", "em", "tôi", "mình", "một", "ly", "cốc",
    "giúp", "với", "nhé", "ạ", "ơi", "có", "gì", "không", "ko",
    "là", "vậy", "tên", "nào", "món", "xin", "hỏi",
}


KEYWORD_ALIASES = {
    "wifi": ["wifi", "mật khẩu", "password"],
    "mở cửa": ["mở cửa", "giờ mở cửa", "mấy giờ"],
    "đóng cửa": ["đóng cửa", "giờ đóng cửa", "mấy giờ"],
    "bạc xỉu": ["bạc xỉu", "bac xiu"],
    "cà phê": ["cà phê", "cafe", "coffee"],
    "trà": ["trà", "tea"],
    "ít ngọt": ["ít ngọt", "less sweet"],
}


def normalize_query(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\sÀ-ỹ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def extract_search_terms(text: str) -> list[str]:
    q = normalize_query(text)

    terms = []

    for canonical, aliases in KEYWORD_ALIASES.items():
        for alias in aliases:
            if alias in q:
                terms.append(canonical)
                break

    tokens = [
        t for t in q.split()
        if len(t) >= 2 and t not in STOPWORDS
    ]

    terms.extend(tokens)

    terms = sorted(set(terms), key=len, reverse=True)

    if not terms:
        terms = [q]

    return terms[:6]


class GraphRetriever:
    def __init__(self) -> None:
        self.settings = get_settings()

    async def retrieve(self, rag_query: RAGQuery) -> RAGResult:
        intent = rag_query.intent
        query = rag_query.query.strip()
        terms = extract_search_terms(query)

        if intent == Intent.ORDER:
            sources = self._retrieve_menu(terms)

        elif intent == Intent.CONSULTANT:
            menu_sources = self._retrieve_menu(terms)
            doc_sources = self._retrieve_docs(terms)

            # Consultant ưu tiên menu
            sources = self._merge_sources(menu_sources, doc_sources)

        elif intent == Intent.FAQ:
            faq_sources = self._retrieve_faq(terms)
            doc_sources = self._retrieve_docs(terms)

            # FAQ ưu tiên FAQ
            sources = self._merge_sources(faq_sources, doc_sources)

        else:
            sources = []

        sources = self._deduplicate_sources(sources)
        sources = sorted(sources, key=lambda x: x.score, reverse=True)
        top_sources = sources[: self.settings.rag_top_k]
        context_text = self._build_context(top_sources)

        logger.info(
            "Retrieval query='%s' terms=%s intent=%s sources=%d",
            query,
            terms,
            intent.value if intent else None,
            len(top_sources),
        )

        return RAGResult(
            query=query,
            sources=top_sources,
            context_text=context_text,
            metadata={"terms": terms},
        )

    async def retrieve_by_vector(
        self,
        rag_query: RAGQuery,
        top_k: int | None = None,
    ) -> RAGResult:
        query = rag_query.query.strip()
        top_k = top_k or rag_query.top_k or self.settings.rag_top_k

        embedding_client = get_embedding_client()
        query_embedding = await embedding_client.embed_text(query)

        if rag_query.intent == Intent.ORDER:
            sources = self._retrieve_menu_by_vector(query_embedding)
        elif rag_query.intent == Intent.FAQ:
            sources = self._retrieve_faq_by_vector(query_embedding)
            sources.extend(self._retrieve_chunks_by_vector(query_embedding))
        elif rag_query.intent == Intent.CONSULTANT:
            sources = self._retrieve_menu_by_vector(query_embedding)
            sources.extend(self._retrieve_faq_by_vector(query_embedding))
            sources.extend(self._retrieve_chunks_by_vector(query_embedding))
        else:
            sources = []

        sources = self._deduplicate_sources(sources)
        sources = sorted(sources, key=lambda x: x.score, reverse=True)
        sources = sources[:top_k]

        return RAGResult(
            query=query,
            sources=sources,
            context_text=self._build_context(sources),
            metadata={
                "retrieval_mode": "vector",
                "top_k": top_k,
            },
        )
    
    async def retrieve_hybrid(
        self,
        rag_query: RAGQuery,
        keyword_weight: float = 0.65,
        vector_weight: float = 0.35,
    ) -> RAGResult:

        keyword_result = await self.retrieve(rag_query)
        vector_result = await self.retrieve_by_vector(rag_query)

        fused_sources = self._late_fusion(
            keyword_sources=keyword_result.sources,
            vector_sources=vector_result.sources,
            keyword_weight=keyword_weight,
            vector_weight=vector_weight,
            top_k=rag_query.top_k or self.settings.rag_top_k,
        )

        if rag_query.intent == Intent.FAQ:
            fused_sources = self._apply_faq_domain_boost(
                query=rag_query.query,
                sources=fused_sources,
            )

        return RAGResult(
            query=rag_query.query,
            sources=fused_sources,
            context_text=self._build_context(fused_sources),
            metadata={
                "retrieval_mode": "hybrid",
                "keyword_count": len(keyword_result.sources),
                "vector_count": len(vector_result.sources),
                "keyword_weight": keyword_weight,
                "vector_weight": vector_weight,
            },
        )

    def _retrieve_menu(self, terms: list[str]) -> list[RetrievedSource]:
        sources = []

        for term in terms:
            rows = neo4j_client.execute_query(
                """
                MATCH (m:MenuItem)
                WHERE
                    toLower(m.name_vi) CONTAINS toLower($term)
                    OR toLower(m.name_en) CONTAINS toLower($term)
                    OR toLower(m.description) CONTAINS toLower($term)
                    OR toLower(m.category) CONTAINS toLower($term)
                    OR EXISTS {
                        MATCH (m)-[:HAS_TAG]->(e:Entity)
                        WHERE toLower(e.name) CONTAINS toLower($term)
                    }
                    OR EXISTS {
                        MATCH (m)-[:HAS_INGREDIENT]->(e:Entity)
                        WHERE toLower(e.name) CONTAINS toLower($term)
                    }

                RETURN
                    m.id as id,
                    m.name_vi as name,
                    m.name_en as name_en,
                    m.description as description,
                    m.price as price,
                    m.category as category,
                    m.size as size
                LIMIT 20
                """,
                {"term": term},
            )

            for row in rows:
                text = (
                    f"{row['name']} | "
                    f"{row['category']} | "
                    f"{row['price']} VND | "
                    f"Size {row['size']} | "
                    f"{row['description']}"
                )

                sources.append(
                    RetrievedSource(
                        source_id=row["id"],
                        source_type=SourceType.MENU,
                        text=text,
                        score=0.95 if term in row["name"].lower() else 0.85,
                        metadata={
                            "price": row["price"],
                            "category": row["category"],
                            "size": row["size"],
                            "matched_term": term,
                        },
                    )
                )

        return sources

    def _retrieve_faq(self, terms: list[str]) -> list[RetrievedSource]:
        sources = []

        for term in terms:
            rows = neo4j_client.execute_query(
                """
                MATCH (f:FAQ)
                WHERE
                    toLower(f.question) CONTAINS toLower($term)
                    OR toLower(f.topic) CONTAINS toLower($term)
                    OR toLower(f.answer) CONTAINS toLower($term)

                RETURN
                    f.id as id,
                    f.topic as topic,
                    f.question as question,
                    f.answer as answer
                LIMIT 20
                """,
                {"term": term},
            )

            for row in rows:
                sources.append(
                    RetrievedSource(
                        source_id=row["id"],
                        source_type=SourceType.FAQ,
                        text=f"Q: {row['question']}\nA: {row['answer']}",
                        score=0.95,
                        metadata={
                            "topic": row["topic"],
                            "matched_term": term,
                        },
                    )
                )

        return sources

    def _retrieve_docs(self, terms: list[str]) -> list[RetrievedSource]:
        sources = []

        for term in terms:
            rows = neo4j_client.execute_query(
                """
                MATCH (c:Chunk)
                OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
                WHERE
                    toLower(c.text) CONTAINS toLower($term)
                    OR toLower(e.name) CONTAINS toLower($term)

                RETURN DISTINCT
                    c.id as id,
                    c.text as text
                LIMIT 20
                """,
                {"term": term},
            )

            for row in rows:
                sources.append(
                    RetrievedSource(
                        source_id=row["id"],
                        source_type=SourceType.DOCUMENT,
                        text=row["text"],
                        score=0.70,
                        metadata={"matched_term": term},
                    )
                )

        return sources

    def _merge_sources(self, *groups: list[RetrievedSource]) -> list[RetrievedSource]:
        merged = []
        for group in groups:
            merged.extend(group)
        return merged

    def _deduplicate_sources(self, sources: list[RetrievedSource]) -> list[RetrievedSource]:
        best_sources = {}

        for src in sources:
            existing = best_sources.get(src.source_id)

            if existing is None or src.score > existing.score:
                best_sources[src.source_id] = src

        return list(best_sources.values())

    def _build_context(self, sources: list[RetrievedSource]) -> str:
        grouped = defaultdict(list)

        for src in sources:
            grouped[src.source_type.value].append(src.text)

        sections = []

        for source_type, values in grouped.items():
            block = "\n".join(values)
            sections.append(f"[{source_type.upper()}]\n{block}")

        return "\n\n".join(sections)

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return dot / (norm_a * norm_b)
    
    def _retrieve_menu_by_vector(self, query_embedding: list[float]) -> list[RetrievedSource]:
        rows = neo4j_client.execute_query(
            """
            MATCH (m:MenuItem)
            WHERE m.embedding IS NOT NULL
            RETURN
                m.id AS id,
                m.name_vi AS name,
                m.name_en AS name_en,
                m.description AS description,
                m.price AS price,
                m.category AS category,
                m.size AS size,
                m.embedding AS embedding
            LIMIT 500
            """
        )

        sources = []
        for row in rows:
            score = self._cosine_similarity(query_embedding, row.get("embedding") or [])
            if score < 0.05:
                continue

            text = (
                f"{row.get('name') or ''} | "
                f"{row.get('category') or ''} | "
                f"{row.get('price') or ''} VND | "
                f"Size {row.get('size') or ''} | "
                f"{row.get('description') or ''}"
            )

            sources.append(
                RetrievedSource(
                    source_id=row["id"],
                    source_type=SourceType.MENU,
                    text=text,
                    score=score,
                    metadata={
                        "retrieval_mode": "vector",
                        "price": row.get("price"),
                        "category": row.get("category"),
                        "size": row.get("size"),
                    },
                )
            )

        return sources
    
    def _retrieve_faq_by_vector(self, query_embedding: list[float]) -> list[RetrievedSource]:
        rows = neo4j_client.execute_query(
            """
            MATCH (f:FAQ)
            WHERE f.embedding IS NOT NULL
            RETURN
                f.id AS id,
                f.topic AS topic,
                f.question AS question,
                f.answer AS answer,
                f.embedding AS embedding
            LIMIT 500
            """
        )

        sources = []
        for row in rows:
            score = self._cosine_similarity(query_embedding, row.get("embedding") or [])
            if score < 0.05:
                continue

            sources.append(
                RetrievedSource(
                    source_id=row["id"],
                    source_type=SourceType.FAQ,
                    text=f"Q: {row.get('question') or ''}\nA: {row.get('answer') or ''}",
                    score=score,
                    metadata={
                        "retrieval_mode": "vector",
                        "topic": row.get("topic"),
                    },
                )
            )

        return sources
    
    def _retrieve_chunks_by_vector(self, query_embedding: list[float]) -> list[RetrievedSource]:
        rows = neo4j_client.execute_query(
            """
            MATCH (c:Chunk)
            WHERE c.embedding IS NOT NULL
            RETURN
                c.id AS id,
                c.text AS text,
                c.embedding AS embedding
            LIMIT 500
            """
        )

        sources = []
        for row in rows:
            score = self._cosine_similarity(query_embedding, row.get("embedding") or [])
            if score < 0.05:
                continue

            sources.append(
                RetrievedSource(
                    source_id=row["id"],
                    source_type=SourceType.DOCUMENT,
                    text=row.get("text") or "",
                    score=score,
                    metadata={
                        "retrieval_mode": "vector",
                    },
                )
            )

        return sources

    def _late_fusion(
        self,
        keyword_sources: list[RetrievedSource],
        vector_sources: list[RetrievedSource],
        keyword_weight: float,
        vector_weight: float,
        top_k: int,
    ) -> list[RetrievedSource]:

        merged: dict[str, RetrievedSource] = {}

        for source in keyword_sources:
            source.score = source.score * keyword_weight

            merged[source.source_id] = source

        for source in vector_sources:
            weighted_score = source.score * vector_weight

            if source.source_id in merged:
                merged[source.source_id].score += weighted_score

                retrieval_modes = merged[source.source_id].metadata.get(
                    "retrieval_modes",
                    []
                )

                retrieval_modes.append("vector")

                merged[source.source_id].metadata[
                    "retrieval_modes"
                ] = retrieval_modes

            else:
                source.score = weighted_score

                source.metadata["retrieval_modes"] = [
                    "vector"
                ]

                merged[source.source_id] = source

        final_sources = list(merged.values())

        final_sources.sort(
            key=lambda x: x.score,
            reverse=True
        )

        return final_sources[:top_k]

    def _apply_faq_domain_boost(
        self,
        query: str,
        sources: list[RetrievedSource],
    ) -> list[RetrievedSource]:

        q = query.lower()

        faq_boost_rules = {
            "wifi": ["wifi", "wi-fi", "internet", "mạng", "mat khau", "mật khẩu", "password"],
            "opening_hours": ["giờ", "mấy giờ", "đóng cửa", "mở cửa", "open", "close"],
            "delivery": ["ship", "giao hàng", "delivery", "mang đi", "take away"],
            "size": ["size", "kích cỡ", "cỡ ly", "s m l"],
        }

        matched_topics = []

        for topic, keywords in faq_boost_rules.items():
            if any(keyword in q for keyword in keywords):
                matched_topics.append(topic)

        if not matched_topics:
            return sources

        for source in sources:
            text = source.text.lower()
            metadata_topic = str(source.metadata.get("topic", "")).lower()

            for topic in matched_topics:
                if topic in metadata_topic or topic in text:
                    source.score += 0.35
                    source.metadata["faq_domain_boost"] = topic

        sources.sort(key=lambda x: x.score, reverse=True)

        return sources

graph_retriever = GraphRetriever()