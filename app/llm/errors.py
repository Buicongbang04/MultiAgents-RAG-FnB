class LLMError(Exception):
    """Lỗi gốc của lớp LLM."""


class LLMUnavailableError(LLMError):
    """Lỗi truyền tải, timeout hoặc model từ chối trả lời."""


class LLMStructuredOutputError(LLMError):
    """Đầu ra không khớp schema yêu cầu."""


class LLMConfigurationError(LLMError):
    """Thiếu hoặc sai cấu hình (vd. không có ANTHROPIC_API_KEY)."""
