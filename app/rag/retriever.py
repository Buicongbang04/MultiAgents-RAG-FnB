import re
from collections import defaultdict

from app.core.config import get_settings
from app.core.constants import Intent, SourceType
from app.core.logging import get_logger
from app.core.schemas import RAGQuery, RAGResult, RetrievedSource
from app.rag.neo4j_client import neo4j_client

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


graph_retriever = GraphRetriever()