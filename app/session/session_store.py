import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.schemas import Message, SessionState

logger = get_logger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


def _history_token_count(history: List[Message]) -> int:
    return sum(_estimate_tokens(m.content) for m in history)


class SessionStore:
    """
    In-memory SessionStore.
    - TTL-based expiration
    - History window (5 recent turns minimum)
    - Auto-summarization khi history vượt 70% context window
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._sessions: Dict[str, SessionState] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._closed = False

    async def start_background_cleanup(self) -> None:
        if self._cleanup_task is None or self._cleanup_task.done():
            self._closed = False
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Session cleanup task started")

    async def close(self) -> None:
        self._closed = True
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("SessionStore closed")

    async def get_or_create(self, session_id: Optional[str] = None) -> SessionState:
        async with self._lock:
            if session_id and session_id in self._sessions:
                session = self._sessions[session_id]
                session.touch()
                return session

            session = SessionState()
            if session_id:
                session.session_id = session_id
            self._sessions[session.session_id] = session
            logger.info("Created new session: %s", session.session_id)
            return session

    async def get(self, session_id: str) -> Optional[SessionState]:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.touch()
            return session

    async def add_user_message(self, session_id: str, content: str) -> SessionState:
        session = await self.get_or_create(session_id)
        async with self._lock:
            session.add_message(role="user", content=content)
            await self._maybe_summarize(session)
            return session

    async def add_assistant_message(self, session_id: str, content: str) -> SessionState:
        session = await self.get_or_create(session_id)
        async with self._lock:
            session.add_message(role="assistant", content=content)
            await self._maybe_summarize(session)
            return session

    async def update_session(self, session: SessionState) -> None:
        async with self._lock:
            session.touch()
            self._sessions[session.session_id] = session

    async def delete(self, session_id: str) -> bool:
        async with self._lock:
            existed = session_id in self._sessions
            self._sessions.pop(session_id, None)
            return existed

    async def list_sessions(self) -> List[SessionState]:
        async with self._lock:
            return list(self._sessions.values())

    async def count(self) -> int:
        async with self._lock:
            return len(self._sessions)

    async def _maybe_summarize(self, session: SessionState) -> None:
        """
        Nếu history vượt 70% SESSION_CONTEXT_TOKEN_LIMIT:
        - Tóm tắt nửa đầu bằng LLM (gọi async, không block)
        - Giữ lại 5 turn gần nhất + summary
        """
        token_limit = self.settings.session_context_token_limit
        trigger_ratio = self.settings.session_summary_trigger_ratio
        threshold = int(token_limit * trigger_ratio)

        if _history_token_count(session.history) < threshold:
            return

        window = self.settings.session_history_window
        keep = session.history[-window:]
        to_summarize = session.history[:-window]

        if not to_summarize:
            return

        logger.info(
            "Auto-summarizing session=%s turns=%d tokens~=%d",
            session.session_id,
            len(to_summarize),
            _history_token_count(to_summarize),
        )

        try:
            summary = await self._summarize_history(to_summarize, session.summary)
            session.summary = summary
            session.history = keep
            logger.info("Summary done session=%s new_length=%d", session.session_id, len(keep))
        except Exception as exc:
            logger.warning("Summarization failed session=%s error=%s", session.session_id, exc)
            max_messages = max(10, window * 4)
            if len(session.history) > max_messages:
                session.history = session.history[-max_messages:]

    async def _summarize_history(
        self,
        messages: List[Message],
        existing_summary: Optional[str],
    ) -> str:
        """Gọi LLM để tóm tắt đoạn hội thoại cũ thành 1 đoạn ngắn."""
        from app.llm import get_llm_client
        from app.llm.base import LLMGenerateRequest

        history_text = "\n".join(
            f"{m.role.upper()}: {m.content}" for m in messages
        )

        prefix = ""
        if existing_summary:
            prefix = f"[Tóm tắt trước đó]: {existing_summary}\n\n"

        llm = get_llm_client()
        response = await llm.generate(
            LLMGenerateRequest(
                system_prompt=(
                    "Bạn là trợ lý tóm tắt hội thoại. "
                    "Tóm tắt ngắn gọn (≤80 từ) nội dung chính của đoạn hội thoại dưới đây, "
                    "giữ lại các thông tin quan trọng như món đã đặt, yêu cầu đặc biệt, "
                    "thông tin FAQ đã được trả lời."
                ),
                user_prompt=(
                    f"{prefix}[Đoạn hội thoại cần tóm tắt]:\n{history_text}"
                ),
                max_tokens=150,
                metadata={"fallback_answer": history_text[:200]},
            )
        )
        return response.text.strip()

    async def _cleanup_loop(self) -> None:
        interval = self.settings.session_cleanup_interval_seconds
        while not self._closed:
            await asyncio.sleep(interval)
            try:
                removed = await self.cleanup_expired()
                if removed:
                    logger.info("Cleaned up %d expired sessions", removed)
            except Exception:
                logger.exception("Session cleanup failed")

    async def cleanup_expired(self) -> int:
        ttl = self.settings.session_ttl_seconds
        now = _utc_now()
        removed = 0
        async with self._lock:
            expired_ids = [
                sid
                for sid, s in self._sessions.items()
                if (now - s.last_active_at).total_seconds() > ttl
            ]
            for sid in expired_ids:
                self._sessions.pop(sid, None)
                removed += 1
        return removed


session_store = SessionStore()
