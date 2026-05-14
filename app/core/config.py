from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central app settings.

    Tất cả thông số quan trọng đi qua .env để sau này chuyển Docker/SGLang/vLLM
    không cần sửa logic code.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # App
    app_name: str = Field(default="multiagent-rag-fnb", alias="APP_NAME")
    app_env: Literal["local", "dev", "staging", "prod"] = Field(default="local", alias="APP_ENV")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Session
    session_ttl_seconds: int = Field(default=1800, alias="SESSION_TTL_SECONDS")
    session_history_window: int = Field(default=5, alias="SESSION_HISTORY_WINDOW")
    session_context_token_limit: int = Field(default=4096, alias="SESSION_CONTEXT_TOKEN_LIMIT")
    session_summary_trigger_ratio: float = Field(default=0.7, alias="SESSION_SUMMARY_TRIGGER_RATIO")
    session_cleanup_interval_seconds: int = Field(default=60, alias="SESSION_CLEANUP_INTERVAL_SECONDS")

    # Queue / concurrency
    router_max_concurrency: int = Field(default=8, alias="ROUTER_MAX_CONCURRENCY")
    generator_max_concurrency: int = Field(default=1, alias="GENERATOR_MAX_CONCURRENCY")
    embedding_max_concurrency: int = Field(default=2, alias="EMBEDDING_MAX_CONCURRENCY")
    reranker_max_concurrency: int = Field(default=1, alias="RERANKER_MAX_CONCURRENCY")
    request_timeout_seconds: int = Field(default=60, alias="REQUEST_TIMEOUT_SECONDS")
    retry_max_attempts: int = Field(default=3, alias="RETRY_MAX_ATTEMPTS")
    retry_base_delay_seconds: float = Field(default=0.3, alias="RETRY_BASE_DELAY_SECONDS")

    # Neo4j
    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", alias="NEO4J_USER")
    neo4j_password: str = Field(default="password", alias="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", alias="NEO4J_DATABASE")

    # LLM serving
    llm_backend: Literal["mock", "sglang", "vllm"] = Field(default="mock", alias="LLM_BACKEND")
    llm_base_url: str = Field(default="http://localhost:30000/v1", alias="LLM_BASE_URL")
    llm_api_key: str = Field(default="EMPTY", alias="LLM_API_KEY")
    generator_model: str = Field(default="qwen2.5-7b-instruct-awq", alias="GENERATOR_MODEL")
    router_model: str = Field(default="router-rule-baseline", alias="ROUTER_MODEL")

    # RAG
    rag_top_k: int = Field(default=5, alias="RAG_TOP_K")
    rag_expansion_window: int = Field(default=1, alias="RAG_EXPANSION_WINDOW")
    rag_relevance_threshold: float = Field(default=0.70, alias="RAG_RELEVANCE_THRESHOLD")

    # Streaming
    stream_token_delay_seconds: float = Field(default=0.01, alias="STREAM_TOKEN_DELAY_SECONDS")
    clause_min_chars: int = Field(default=12, alias="CLAUSE_MIN_CHARS")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings instance."""

    return Settings()