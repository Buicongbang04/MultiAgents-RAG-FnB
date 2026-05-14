from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from app.core.constants import AgentName, Intent, Language, SourceType, StreamEventType, VALID_INTENTS


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:16]}"

# -------------------------
# Common API schemas
# -------------------------
class ChatRequest(BaseModel):
    """Input cho /chat và /chat/stream."""

    text: str = Field(..., min_length=1, description="User query text")
    session_id: Optional[str] = Field(default=None, description="Existing session_id or None to create new")
    language: Optional[Language] = Field(default=None, description="Optional client-provided language")
    stream: bool = Field(default=False)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text must not be empty")
        return value


class ChatResponse(BaseModel):
    """Response non-stream."""

    session_id: str
    intent: Intent
    agent: AgentName
    answer: str
    language: Language = Language.UNKNOWN
    sources: List["RetrievedSource"] = Field(default_factory=list)
    latency_ms: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: str
    message: str
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

# -------------------------
# Router schemas
# -------------------------
class RouterInput(BaseModel):
    session_id: str
    text: str
    language: Optional[Language] = None
    history: List["Message"] = Field(default_factory=list)


class RouterOutput(BaseModel):
    """
    Router output nội bộ.

    Public requirement vẫn đảm bảo có JSON dạng {"action": "<intent>"}.
    """

    action: Intent
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    language: Language = Language.UNKNOWN
    raw_output: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_required_json(self) -> Dict[str, str]:
        return {"action": self.action.value}

    @field_validator("action", mode="before")
    @classmethod
    def validate_action(cls, value: Any) -> Any:
        if isinstance(value, Intent):
            return value
        if isinstance(value, str) and value in VALID_INTENTS:
            return value
        raise ValueError(f"Invalid router action: {value}")

# -------------------------
# Session schemas
# -------------------------
class Message(BaseModel):
    role: str = Field(..., description="user | assistant | system | tool")
    content: str
    created_at: datetime = Field(default_factory=utc_now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SessionState(BaseModel):
    session_id: str = Field(default_factory=lambda: new_id("sess"))
    created_at: datetime = Field(default_factory=utc_now)
    last_active_at: datetime = Field(default_factory=utc_now)
    language: Language = Language.UNKNOWN
    history: List[Message] = Field(default_factory=list)
    summary: Optional[str] = None
    last_intent: Optional[Intent] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def touch(self) -> None:
        self.last_active_at = utc_now()

    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.history.append(
            Message(
                role=role,
                content=content,
                metadata=metadata or {},
            )
        )
        self.touch()

    def recent_history(self, window: int = 5) -> List[Message]:
        if window <= 0:
            return []
        return self.history[-window:]

# -------------------------
# Graph/RAG schemas
# -------------------------
class MenuItem(BaseModel):
    id: str = Field(default_factory=lambda: new_id("menu"))
    name_vi: str
    name_en: Optional[str] = None
    price: int = Field(..., ge=0, description="Price in VND")
    size: Optional[str] = None
    category: str
    ingredients: List[str] = Field(default_factory=list)
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    available: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def searchable_text(self) -> str:
        parts = [
            self.name_vi,
            self.name_en or "",
            self.category,
            self.description or "",
            " ".join(self.ingredients),
            " ".join(self.tags),
        ]
        return " | ".join([p for p in parts if p]).strip()


class Category(BaseModel):
    id: str = Field(default_factory=lambda: new_id("cat"))
    name_vi: str
    name_en: Optional[str] = None


class Entity(BaseModel):
    id: str = Field(default_factory=lambda: new_id("ent"))
    name: str
    type: str = Field(default="generic")
    normalized_name: Optional[str] = None

    @field_validator("normalized_name", mode="before")
    @classmethod
    def default_normalized_name(cls, value: Optional[str], info: Any) -> Optional[str]:
        return value


class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: new_id("chunk"))
    source_type: SourceType
    source_file: Optional[str] = None
    text: str
    language: Language = Language.UNKNOWN
    chunk_index: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievedSource(BaseModel):
    """Một kết quả retrieval được đưa vào Agent/Generator."""

    source_id: str
    source_type: SourceType
    text: str
    score: float = Field(default=0.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RAGQuery(BaseModel):
    query: str
    intent: Optional[Intent] = None
    language: Language = Language.UNKNOWN
    top_k: int = 5
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def copy_with_query(self, query: str) -> "RAGQuery":
        return self.model_copy(update={"query": query})


class RAGResult(BaseModel):
    query: str
    sources: List[RetrievedSource] = Field(default_factory=list)
    context_text: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def has_context(self) -> bool:
        return bool(self.sources and self.context_text.strip())
    
# -------------------------
# Agent schemas
# -------------------------
class AgentInput(BaseModel):
    session_id: str
    text: str
    intent: Intent
    language: Language = Language.UNKNOWN
    history: List[Message] = Field(default_factory=list)
    rag_result: Optional[RAGResult] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentOutput(BaseModel):
    session_id: str
    intent: Intent
    agent: AgentName
    answer: str
    language: Language = Language.UNKNOWN
    sources: List[RetrievedSource] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

# -------------------------
# LLM schemas
# -------------------------
class LLMMessage(BaseModel):
    role: str
    content: str


class LLMRequest(BaseModel):
    model: str
    messages: List[LLMMessage]
    temperature: float = 0.2
    max_tokens: int = 512
    stream: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    text: str
    model: str
    latency_ms: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

# -------------------------
# Streaming schemas
# -------------------------
class StreamEvent(BaseModel):
    type: StreamEventType
    data: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

    def to_sse(self) -> str:
        """
        Convert event to Server-Sent Events format.

        [DONE] sẽ được xử lý riêng ở streaming module sau.
        """
        import orjson

        payload = orjson.dumps(
            {
                "type": self.type.value,
                "data": self.data,
                "created_at": self.created_at.isoformat(),
            }
        ).decode("utf-8")
        return f"data: {payload}\n\n"


# Pydantic forward refs
ChatResponse.model_rebuild()
RouterInput.model_rebuild()
SessionState.model_rebuild()
AgentInput.model_rebuild()