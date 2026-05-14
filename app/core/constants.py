from enum import Enum


class Intent(str, Enum):
    """4 intent chính theo yêu cầu đề bài."""

    ORDER = "order"
    CONSULTANT = "consultant"
    FAQ = "faq"
    IGNORE = "ignore"


class Language(str, Enum):
    """Ngôn ngữ MVP. Có thể mở rộng thêm sau."""

    VI = "vi"
    EN = "en"
    UNKNOWN = "unknown"


class AgentName(str, Enum):
    """Tên các agent trong hệ thống."""

    ROUTER = "router"
    ORDER = "order_agent"
    CONSULTANT = "consultant_agent"
    FAQ = "faq_agent"
    IGNORE = "ignore_handler"


class SourceType(str, Enum):
    """Loại nguồn dữ liệu trong Graph RAG."""

    MENU = "menu"
    FAQ = "faq"
    DOCUMENT = "document"


class StreamEventType(str, Enum):
    """Event type cho SSE streaming."""

    TOKEN = "token"
    CLAUSE = "clause"
    DONE = "done"
    ERROR = "error"


VALID_INTENTS = {intent.value for intent in Intent}


# Mapping label đúng yêu cầu A1 Router SFT sau này.
INTENT_TO_LABEL = {
    Intent.ORDER.value: 0,
    Intent.CONSULTANT.value: 1,
    Intent.FAQ.value: 2,
    Intent.IGNORE.value: 3,
}

LABEL_TO_INTENT = {v: k for k, v in INTENT_TO_LABEL.items()}