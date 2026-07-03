import re
from collections import defaultdict
import math

from app.core.config import get_settings
from app.core.constants import Intent, SourceType
from app.core.logging import get_logger
from app.core.schemas import RAGQuery, RAGResult, RetrievedSource
from app.rag.neo4j_client import neo4j_client
from app.rag.embedding_client import get_embedding_client
from app.rag.reranker import get_reranker

logger = get_logger(__name__)

# Warn once per index instead of on every request, so a missing vector index is
# visible in logs without spamming.
_warned_vector_indexes: set[str] = set()


def _warn_vector_index_fallback(index_name: str, exc: Exception) -> None:
    if index_name not in _warned_vector_indexes:
        _warned_vector_indexes.add(index_name)
        logger.warning(
            "Vector index '%s' unavailable (%s). Falling back to full-scan cosine; "
            "run Neo4jClient.ensure_vector_indexes() to enable the fast path.",
            index_name, exc,
        )


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

    async def retrieve_hybrid_with_graph_expansion(
        self,
        rag_query: RAGQuery,
        keyword_weight: float = 0.65,
        vector_weight: float = 0.35,
        expansion_weight: float = 0.75,
    ) -> RAGResult:

        hybrid_result = await self.retrieve_hybrid(
            rag_query=rag_query,
            keyword_weight=keyword_weight,
            vector_weight=vector_weight,
        )

        expanded_sources = self._expand_graph_sources(
            sources=hybrid_result.sources,
            expansion_weight=expansion_weight,
        )

        final_sources = self._deduplicate_sources(
            hybrid_result.sources + expanded_sources
        )

        # Lightweight lexical rerank
        final_sources = self._lightweight_rerank(
            query=rag_query.query,
            intent=rag_query.intent,
            sources=final_sources,
        )

        top_k = rag_query.top_k or self.settings.rag_top_k

        # BGE reranker (nếu được bật)
        reranker = get_reranker()
        final_sources = reranker.rerank(
            query=rag_query.query,
            sources=final_sources,
            top_k=top_k,
            threshold=self.settings.reranker_threshold,
        )

        return RAGResult(
            query=rag_query.query,
            sources=final_sources,
            context_text=self._build_context(final_sources),
            metadata={
                "retrieval_mode": "hybrid_graph_expansion",
                "hybrid_count": len(hybrid_result.sources),
                "expanded_count": len(expanded_sources),
                "final_count": len(final_sources),
                "keyword_weight": keyword_weight,
                "vector_weight": vector_weight,
                "expansion_weight": expansion_weight,
                "reranker": type(reranker).__name__,
            },
        )

    async def retrieve_auto(
        self,
        rag_query: RAGQuery,
    ) -> RAGResult:

        mode = getattr(self.settings, "rag_retrieval_mode", "keyword")

        if mode == "keyword":
            return await self.retrieve(rag_query)

        if mode == "vector":
            return await self.retrieve_by_vector(rag_query)

        if mode == "hybrid":
            if rag_query.intent == Intent.FAQ:
                return await self.retrieve_hybrid(
                    rag_query,
                    keyword_weight=0.45,
                    vector_weight=0.55,
                )

            return await self.retrieve_hybrid(rag_query)

        if mode == "hybrid_graph":
            if rag_query.intent == Intent.FAQ:
                return await self.retrieve_hybrid_with_graph_expansion(
                    rag_query,
                    keyword_weight=0.45,
                    vector_weight=0.55,
                )

            return await self.retrieve_hybrid_with_graph_expansion(rag_query)

        logger.warning(
            "Unknown RAG_RETRIEVAL_MODE=%s. Falling back to keyword retrieval.",
            mode,
        )

        return await self.retrieve(rag_query)

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
        # Thử dùng Neo4j vector index trước (nhanh hơn ~10x so với fetch 500 rows)
        try:
            rows = neo4j_client.execute_query(
                """
                CALL db.index.vector.queryNodes('menu_embedding', $top_k, $embedding)
                YIELD node AS m, score
                RETURN
                    m.id AS id,
                    m.name_vi AS name,
                    m.name_en AS name_en,
                    m.description AS description,
                    m.price AS price,
                    m.category AS category,
                    m.size AS size,
                    score
                """,
                {"embedding": query_embedding, "top_k": 20},
            )
            if rows:
                return [
                    RetrievedSource(
                        source_id=r["id"],
                        source_type=SourceType.MENU,
                        text=(
                            f"{r.get('name') or ''} | "
                            f"{r.get('category') or ''} | "
                            f"{r.get('price') or ''} VND | "
                            f"Size {r.get('size') or ''} | "
                            f"{r.get('description') or ''}"
                        ),
                        score=float(r.get("score", 0.0)),
                        metadata={
                            "retrieval_mode": "vector_index",
                            "price": r.get("price"),
                            "category": r.get("category"),
                            "size": r.get("size"),
                        },
                    )
                    for r in rows
                ]
        except Exception as exc:
            _warn_vector_index_fallback("menu_embedding", exc)

        # Fallback: fetch rows + Python cosine
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
            sources.append(
                RetrievedSource(
                    source_id=row["id"],
                    source_type=SourceType.MENU,
                    text=(
                        f"{row.get('name') or ''} | "
                        f"{row.get('category') or ''} | "
                        f"{row.get('price') or ''} VND | "
                        f"Size {row.get('size') or ''} | "
                        f"{row.get('description') or ''}"
                    ),
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
        try:
            # Use the FAQ-specific index. Querying the shared 'chunk_embedding' index
            # and filtering `WHERE f:FAQ` *after* top-k lets Chunk nodes crowd out FAQs.
            rows = neo4j_client.execute_query(
                """
                CALL db.index.vector.queryNodes('faq_embedding', $top_k, $embedding)
                YIELD node AS f, score
                RETURN f.id AS id, f.topic AS topic, f.question AS question,
                       f.answer AS answer, score
                """,
                {"embedding": query_embedding, "top_k": 20},
            )
            if rows:
                return [
                    RetrievedSource(
                        source_id=r["id"],
                        source_type=SourceType.FAQ,
                        text=f"Q: {r.get('question') or ''}\nA: {r.get('answer') or ''}",
                        score=float(r.get("score", 0.0)),
                        metadata={"retrieval_mode": "vector_index", "topic": r.get("topic")},
                    )
                    for r in rows
                ]
        except Exception as exc:
            _warn_vector_index_fallback("faq_embedding", exc)

        rows = neo4j_client.execute_query(
            """
            MATCH (f:FAQ)
            WHERE f.embedding IS NOT NULL
            RETURN f.id AS id, f.topic AS topic, f.question AS question,
                   f.answer AS answer, f.embedding AS embedding
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
                    metadata={"retrieval_mode": "vector", "topic": row.get("topic")},
                )
            )
        return sources

    def _retrieve_chunks_by_vector(self, query_embedding: list[float]) -> list[RetrievedSource]:
        try:
            rows = neo4j_client.execute_query(
                """
                CALL db.index.vector.queryNodes('chunk_embedding', $top_k, $embedding)
                YIELD node AS c, score
                WHERE c:Chunk
                RETURN c.id AS id, c.text AS text, score
                """,
                {"embedding": query_embedding, "top_k": 20},
            )
            if rows:
                return [
                    RetrievedSource(
                        source_id=r["id"],
                        source_type=SourceType.DOCUMENT,
                        text=r.get("text") or "",
                        score=float(r.get("score", 0.0)),
                        metadata={"retrieval_mode": "vector_index"},
                    )
                    for r in rows
                ]
        except Exception as exc:
            _warn_vector_index_fallback("chunk_embedding", exc)

        rows = neo4j_client.execute_query(
            """
            MATCH (c:Chunk)
            WHERE c.embedding IS NOT NULL
            RETURN c.id AS id, c.text AS text, c.embedding AS embedding
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
                    metadata={"retrieval_mode": "vector"},
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

    def _expand_graph_sources(
        self,
        sources: list[RetrievedSource],
        expansion_weight: float = 0.75,
    ) -> list[RetrievedSource]:

        expanded: list[RetrievedSource] = []

        for source in sources:

            if source.source_type == SourceType.DOCUMENT:
                expanded.extend(
                    self._expand_chunk_neighbors(
                        source=source,
                        expansion_weight=expansion_weight,
                    )
                )

                expanded.extend(
                    self._expand_chunk_entities(
                        source=source,
                        expansion_weight=expansion_weight,
                    )
                )

            elif source.source_type == SourceType.FAQ:
                expanded.extend(
                    self._expand_faq_entities(
                        source=source,
                        expansion_weight=expansion_weight,
                    )
                )

            elif source.source_type == SourceType.MENU:
                expanded.extend(
                    self._expand_menu_category(
                        source=source,
                        expansion_weight=expansion_weight,
                    )
                )

        return expanded
    
    def _expand_chunk_neighbors(
        self,
        source: RetrievedSource,
        expansion_weight: float,
    ) -> list[RetrievedSource]:

        rows = neo4j_client.execute_query(
            """
            MATCH (c:Chunk {id: $source_id})
            OPTIONAL MATCH (prev:Chunk)-[:NEXT]->(c)
            OPTIONAL MATCH (c)-[:NEXT]->(next:Chunk)
            WITH collect(prev) + collect(next) AS neighbors
            UNWIND neighbors AS n
            WITH n
            WHERE n IS NOT NULL
            RETURN
                n.id AS id,
                n.text AS text
            LIMIT 4
            """,
            {"source_id": source.source_id},
        )

        expanded = []

        for row in rows:
            expanded.append(
                RetrievedSource(
                    source_id=row["id"],
                    source_type=SourceType.DOCUMENT,
                    text=row.get("text") or "",
                    score=source.score * expansion_weight,
                    metadata={
                        "retrieval_mode": "graph_expansion",
                        "expanded_from": source.source_id,
                        "relation": "NEXT_PREV",
                    },
                )
            )

        return expanded
    
    def _expand_chunk_entities(
        self,
        source: RetrievedSource,
        expansion_weight: float,
    ) -> list[RetrievedSource]:

        rows = neo4j_client.execute_query(
            """
            MATCH (c:Chunk {id: $source_id})-[:MENTIONS]->(e:Entity)
            RETURN
                e.key AS id,
                e.name AS name,
                e.type AS type
            LIMIT 10
            """,
            {"source_id": source.source_id},
        )

        expanded = []

        for row in rows:
            text = f"Entity: {row.get('name') or ''} | Type: {row.get('type') or ''}"

            entity_id = row.get('id')
            if not entity_id:
                continue

            expanded.append(
                RetrievedSource(
                    source_id=entity_id,
                    source_type=SourceType.DOCUMENT,
                    text=text,
                    score=source.score * expansion_weight * 0.85,
                    metadata={
                        "retrieval_mode": "graph_expansion",
                        "expanded_from": source.source_id,
                        "relation": "MENTIONS",
                        "entity_type": row.get("type"),
                    },
                )
            )

        return expanded

    def _expand_faq_entities(
        self,
        source: RetrievedSource,
        expansion_weight: float,
    ) -> list[RetrievedSource]:

        rows = neo4j_client.execute_query(
            """
            MATCH (f:FAQ {id: $source_id})-[:MENTIONS]->(e:Entity)
            RETURN
                e.key AS id,
                e.name AS name,
                e.type AS type
            LIMIT 10
            """,
            {"source_id": source.source_id},
        )

        expanded = []

        for row in rows:
            text = f"Entity: {row.get('name') or ''} | Type: {row.get('type') or ''}"

            entity_id = row.get('id')
            if not entity_id:
                continue

            expanded.append(
                RetrievedSource(
                    source_id=entity_id,
                    source_type=SourceType.FAQ,
                    text=text,
                    score=source.score * expansion_weight * 0.85,
                    metadata={
                        "retrieval_mode": "graph_expansion",
                        "expanded_from": source.source_id,
                        "relation": "MENTIONS",
                        "entity_type": row.get("type"),
                    },
                )
            )

        return expanded
    
    def _expand_menu_category(
        self,
        source: RetrievedSource,
        expansion_weight: float,
    ) -> list[RetrievedSource]:

        rows = neo4j_client.execute_query(
            """
            MATCH (m:MenuItem {id: $source_id})-[:BELONGS_TO]->(cat:Category)<-[:BELONGS_TO]-(other:MenuItem)
            WHERE other.id <> m.id
            RETURN
                other.id AS id,
                other.name_vi AS name,
                other.name_en AS name_en,
                other.description AS description,
                other.price AS price,
                other.category AS category,
                other.size AS size
            LIMIT 5
            """,
            {"source_id": source.source_id},
        )

        expanded = []

        for row in rows:
            text = (
                f"{row.get('name') or ''} | "
                f"{row.get('category') or ''} | "
                f"{row.get('price') or ''} VND | "
                f"Size {row.get('size') or ''} | "
                f"{row.get('description') or ''}"
            )

            expanded.append(
                RetrievedSource(
                    source_id=row["id"],
                    source_type=SourceType.MENU,
                    text=text,
                    score=source.score * expansion_weight * 0.65,
                    metadata={
                        "retrieval_mode": "graph_expansion",
                        "expanded_from": source.source_id,
                        "relation": "BELONGS_TO_CATEGORY",
                        "category": row.get("category"),
                        "price": row.get("price"),
                        "size": row.get("size"),
                    },
                )
            )

        return expanded

    def _lightweight_rerank(
        self,
        query: str,
        intent: Intent,
        sources: list[RetrievedSource],
    ) -> list[RetrievedSource]:

        q = query.lower()

        for source in sources:
            text = source.text.lower()
            metadata = source.metadata or {}

            lexical_overlap = self._lexical_overlap_score(q, text)

            source.score += lexical_overlap * 0.15

            retrieval_mode = metadata.get("retrieval_mode", "")

            if retrieval_mode == "graph_expansion":
                source.score *= 0.85

            if intent == Intent.ORDER and source.source_type == SourceType.MENU:
                source.score *= 1.10

            if intent == Intent.FAQ and source.source_type == SourceType.FAQ:
                source.score *= 1.10

            if intent == Intent.CONSULTANT and source.source_type in [
                SourceType.MENU,
                SourceType.DOCUMENT,
            ]:
                source.score *= 1.05

        sources.sort(key=lambda x: x.score, reverse=True)

        return sources

    def _lexical_overlap_score(
        self,
        query: str,
        text: str,
    ) -> float:

        query_tokens = self._simple_tokenize(query)
        text_tokens = self._simple_tokenize(text)

        if not query_tokens or not text_tokens:
            return 0.0

        overlap = query_tokens.intersection(text_tokens)

        return len(overlap) / max(len(query_tokens), 1)

    def _simple_tokenize(self, text: str) -> set[str]:
        stopwords = {
            "là",
            "gì",
            "của",
            "quán",
            "cho",
            "anh",
            "chị",
            "em",
            "tôi",
            "một",
            "ly",
            "có",
            "không",
            "the",
            "a",
            "an",
            "is",
            "are",
            "what",
        }

        cleaned = (
            text.lower()
            .replace("?", " ")
            .replace(".", " ")
            .replace(",", " ")
            .replace(":", " ")
            .replace(";", " ")
            .replace("|", " ")
            .replace("-", " ")
            .replace("_", " ")
        )

        tokens = {
            token.strip()
            for token in cleaned.split()
            if token.strip() and token.strip() not in stopwords
        }

        return tokens
    

graph_retriever = GraphRetriever()