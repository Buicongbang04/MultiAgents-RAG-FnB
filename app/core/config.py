"""
Cấu hình tập trung.

- configs/app.yaml      → Settings  (get_settings())   tham số runtime
- configs/lexicon.yaml  → Lexicon   (get_lexicon())    từ khoá / regex miền
- configs/training.yaml → Training  (get_training_config()) siêu tham số fine-tune

Ưu tiên cho Settings: biến môi trường > .env > app.yaml.
Key lồng nhau override bằng "__": LLM__BACKEND=sglang, NEO4J__PASSWORD=...

Các field không có default: YAML là nguồn duy nhất, thiếu key thì báo lỗi ngay
khi khởi động thay vì âm thầm dùng một giá trị ẩn trong code.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

import yaml
from pydantic import BaseModel, ConfigDict
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(os.getenv("APP_CONFIG_DIR", PROJECT_ROOT / "configs"))


class _Section(BaseModel):
    # Bắt lỗi gõ sai key trong YAML (vd. "treshold") thay vì bỏ qua.
    model_config = ConfigDict(extra="forbid", frozen=True)


# ── App / server ──────────────────────────────────────────────────────────────
class AppSection(_Section):
    name: str
    title: str
    version: str
    env: Literal["local", "dev", "staging", "prod"]
    log_level: str


class ServerSection(_Section):
    host: str
    port: int
    ui_port: int
    neo4j_startup_retries: int
    neo4j_startup_retry_delay_seconds: float


class RateLimitSection(_Section):
    max_requests: int
    window_seconds: int
    exempt_path_prefixes: List[str]


class HealthSection(_Section):
    llm_timeout_seconds: float


class SessionSection(_Section):
    ttl_seconds: int
    history_window: int
    context_token_limit: int
    summary_trigger_ratio: float
    cleanup_interval_seconds: int
    chars_per_token: int
    summary_max_tokens: int
    summary_max_words: int
    summary_fallback_chars: int
    hard_trim_min_messages: int
    hard_trim_window_multiplier: int


class ConversationSection(_Section):
    followup_enabled: bool
    followup_max_words: int
    followup_max_turn_gap: int
    followup_intents: List[str]
    refine_intents: List[str]
    max_query_segments: int
    rewrite_with_llm: bool
    rewrite_max_tokens: int
    rewrite_temperature: float


class OrderDialogueSection(_Section):
    enabled: bool
    pending_on_other_intents: Literal["remind", "clear"]


class QueueConcurrency(_Section):
    router: int
    generator: int
    embedding: int
    reranker: int


class QueueSection(_Section):
    request_timeout_seconds: int
    retry_max_attempts: int
    retry_base_delay_seconds: float
    max_concurrency: QueueConcurrency


class Neo4jSection(_Section):
    uri: str
    user: str
    password: str
    database: str


# ── LLM ───────────────────────────────────────────────────────────────────────
class LLMServerSection(_Section):
    host: str
    port: int
    context_length: int
    mem_fraction_static: float
    max_running_requests: int
    chunked_prefill_size: int
    schedule_policy: str


class LLMSection(_Section):
    backend: Literal["mock", "sglang", "vllm"]
    base_url: str
    api_key: str
    model: str
    temperature: float
    max_tokens: int
    timeout_seconds: float
    stream_timeout_seconds: float
    server: LLMServerSection


# ── Router / intent extractor ─────────────────────────────────────────────────
class HFModelSection(_Section):
    base_model: str
    adapter_dir: str
    merged_model_dir: str
    device: str
    max_new_tokens: int


class RouterLLMSection(_Section):
    temperature: float
    max_tokens: int
    timeout_seconds: float
    confidence: float


class RouterRulesSection(_Section):
    base_scores: Dict[str, float]
    extra_match_bonus: float
    max_extra_bonus: float
    quantity_menu_bonus: float
    menu_item_alone_min_score: float
    preference_question_bonus: float
    short_noise_max_chars: int
    short_noise_bonus: float
    order_priority_threshold: float
    no_match_score: float
    max_confidence: float


class RouterSection(_Section):
    backend: Literal["rule_based", "hf_lora", "hf_merged"]
    llm: RouterLLMSection
    hf: HFModelSection
    rules: RouterRulesSection


class IntentExtractorHFSection(HFModelSection):
    confidence: float


class IntentExtractorSection(_Section):
    enabled: bool
    backend: Literal["rule_based", "hf_lora", "hf_merged"]
    hf: IntentExtractorHFSection


# ── Embedding / reranker ──────────────────────────────────────────────────────
class EmbeddingSection(_Section):
    backend: Literal["mock", "sentence_transformers"]
    model: str
    dim: int
    device: str
    batch_size: int


class RerankerSection(_Section):
    backend: Literal["bge", "null"]
    model: str
    device: str
    threshold: float


# ── RAG ───────────────────────────────────────────────────────────────────────
class RAGQuerySection(_Section):
    max_terms: int
    min_token_length: int


class RAGKeywordSection(_Section):
    limit_per_term: int
    score_menu_name_match: float
    score_menu_other_match: float
    score_faq: float
    score_document: float
    coverage_base: float
    coverage_weight: float


class RAGVectorIndexes(_Section):
    menu: str
    faq: str
    chunk: str


class RAGVectorSection(_Section):
    top_k: int
    min_cosine: float
    fallback_scan_limit: int
    indexes: RAGVectorIndexes


class FusionWeights(_Section):
    keyword_weight: float
    vector_weight: float


class RAGFusionSection(_Section):
    default: FusionWeights
    faq: FusionWeights


class RAGExpansionSection(_Section):
    weight: float
    entity_factor: float
    category_factor: float
    neighbor_limit: int
    entity_limit: int
    category_limit: int


class RAGLightweightRerankSection(_Section):
    lexical_overlap_weight: float
    graph_expansion_penalty: float
    order_menu_boost: float
    faq_faq_boost: float
    consultant_menu_document_boost: float


class RAGContextSection(_Section):
    consultant_max_menu_items: int
    consultant_max_supporting: int
    consultant_fallback_items: int
    faq_max_items: int


class RAGSection(_Section):
    retrieval_mode: Literal["keyword", "vector", "hybrid", "hybrid_graph"]
    top_k: int
    candidate_pool_size: int
    rerank_clean_query_intents: List[str]
    query: RAGQuerySection
    keyword: RAGKeywordSection
    vector: RAGVectorSection
    fusion: RAGFusionSection
    faq_domain_boost: float
    expansion: RAGExpansionSection
    lightweight_rerank: RAGLightweightRerankSection
    context: RAGContextSection


# ── Cache / streaming / UI ────────────────────────────────────────────────────
class ExactCacheSection(_Section):
    enabled: bool
    ttl_seconds: int
    max_size: int


class SemanticCacheSection(_Section):
    enabled: bool
    ttl_seconds: int
    max_size: int
    default_threshold: float
    thresholds: Dict[str, float]


class CacheSection(_Section):
    enabled: bool
    cacheable_intents: List[str]
    skip_intents: List[str]
    exact: ExactCacheSection
    semantic: SemanticCacheSection


class StreamingSection(_Section):
    clause_min_chars: int


class TypingDelaySection(_Section):
    newline: float
    punctuation: float
    bullet: float
    long_word: float
    word: float
    first_word: float


class UISection(_Section):
    api_base_url: str
    request_timeout_seconds: float
    long_word_chars: int
    typing_delay_seconds: TypingDelaySection


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
        frozen=True,
    )

    app: AppSection
    server: ServerSection
    rate_limit: RateLimitSection
    health: HealthSection
    session: SessionSection
    conversation: ConversationSection
    order_dialogue: OrderDialogueSection
    queue: QueueSection
    neo4j: Neo4jSection
    llm: LLMSection
    router: RouterSection
    intent_extractor: IntentExtractorSection
    embedding: EmbeddingSection
    reranker: RerankerSection
    rag: RAGSection
    cache: CacheSection
    streaming: StreamingSection
    ui: UISection

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        yaml_settings = YamlConfigSettingsSource(
            settings_cls,
            yaml_file=CONFIG_DIR / "app.yaml",
            yaml_file_encoding="utf-8",
        )
        return (init_settings, env_settings, dotenv_settings, yaml_settings, file_secret_settings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# ── Lexicon ───────────────────────────────────────────────────────────────────
class LanguageLexicon(_Section):
    vi_diacritics_pattern: str
    english_word_pattern: str
    rule_router_markers: Dict[str, List[str]]
    intent_extractor_vi_markers: List[str]
    hf_router_vi_markers: List[str]


class RouterRulesLexicon(_Section):
    keywords: Dict[str, List[str]]
    question_markers: List[str]
    menu_item_hints: List[str]
    preference_words: List[str]
    quantity_pattern: str


class RetrievalLexicon(_Section):
    stopwords: List[str]
    keyword_aliases: Dict[str, List[str]]
    faq_boost_rules: Dict[str, List[str]]
    lexical_stopwords: List[str]


class SemanticCacheLexicon(_Section):
    domain_tags: Dict[str, List[str]]
    faq_alias_tags: List[str]


class LocalizedAction(_Section):
    action: str
    cache_key: str


class FAQExtractionRule(_Section):
    name: str
    pattern: str
    confidence: float
    vi: LocalizedAction
    en: LocalizedAction
    only_for_intent: Optional[str] = None  # None = áp dụng cho mọi intent


class ConsultantPreference(_Section):
    name: str
    pattern: str


class ExtractorConfidence(_Section):
    consultant_generic: float
    consultant_with_preferences: float
    order: float
    ignore: float
    fallback: float


class IntentExtractorLexicon(_Section):
    subject_patterns: List[str]
    english_subject_pattern: str
    context_patterns: List[str]
    filler_patterns: List[str]
    filler_words_pattern: str
    faq_rules: List[FAQExtractionRule]
    recommend_pattern: str
    budget_pattern: str
    budget_preference: str
    consultant_preferences: List[ConsultantPreference]
    preference_overrides: Dict[str, str]
    consultant_generic: Dict[str, str]
    consultant_prefix: Dict[str, str]
    order_strip_pattern: str
    confidence: ExtractorConfidence


class ConversationLexicon(_Section):
    followup_markers: List[str]
    greeting_words: List[str]


class OrderLexicon(_Section):
    size_patterns: Dict[str, List[str]]
    trailing_particles: List[str]
    bare_size_words: Dict[str, List[str]]
    quantity_digit_pattern: str
    quantity_word_pattern: str
    quantity_words: Dict[str, int]
    generic_size_pattern: str
    generic_item_words: List[str]
    non_item_words: List[str]
    availability_patterns: List[str]
    generic_availability_pattern: str
    max_item_choices: int


class OrderDialogueLexicon(_Section):
    particles: List[str]
    affirm: List[str]
    negate: List[str]
    checkout: List[str]
    view_cart: List[str]
    cancel_order: List[str]
    remove_pattern: str


class Lexicon(_Section):
    language: LanguageLexicon
    router_rules: RouterRulesLexicon
    retrieval: RetrievalLexicon
    semantic_cache: SemanticCacheLexicon
    intent_extractor: IntentExtractorLexicon
    conversation: ConversationLexicon
    order: OrderLexicon
    order_dialogue: OrderDialogueLexicon


def _load_yaml(name: str) -> Dict[str, Any]:
    path = CONFIG_DIR / name
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def get_lexicon() -> Lexicon:
    return Lexicon.model_validate(_load_yaml("lexicon.yaml"))


# ── Training ──────────────────────────────────────────────────────────────────
class SFTConfig(_Section):
    train_path: str
    val_path: str
    test_path: str
    max_length: int
    use_4bit: bool
    quantization: Dict[str, Any]
    lora: Dict[str, Any]
    training_args: Dict[str, Any]


class MergeConfig(_Section):
    dtype: str
    device_map: str


class TrainingConfig(_Section):
    router_sft: SFTConfig
    intent_extractor_sft: SFTConfig
    merge: MergeConfig


@lru_cache(maxsize=1)
def get_training_config() -> TrainingConfig:
    return TrainingConfig.model_validate(_load_yaml("training.yaml"))
