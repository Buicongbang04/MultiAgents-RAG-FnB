import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.schemas import Message, SessionState

logger = get_logger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SessionStore:
    """
    In-memory SessionStore cho MVP.

    Production phase sau có thể thay bằng Redis mà không đổi interface chính.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._sessions: Dict[str, SessionState] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._closed = False

    async def start_background_cleanup(self) -> None:
        """Start background cleanup task nếu chưa chạy."""

        if self._cleanup_task is None or self._cleanup_task.done():
            self._closed = False
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Session cleanup task started")

    async def close(self) -> None:
        """Stop background cleanup task."""

        self._closed = True
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("SessionStore closed")

    async def get_or_create(self, session_id: Optional[str] = None) -> SessionState:
        """
        Lấy session nếu tồn tại, nếu không thì tạo mới.

        Nếu client truyền session_id không tồn tại, hệ thống tạo session mới với id đó
        để dễ debug/demo.
        """

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
            self._trim_history_if_needed(session)
            return session

    async def add_assistant_message(self, session_id: str, content: str) -> SessionState:
        session = await self.get_or_create(session_id)
        async with self._lock:
            session.add_message(role="assistant", content=content)
            self._trim_history_if_needed(session)
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
        
    def _trim_history_if_needed(self, session: SessionState) -> None:
        """
        MVP: giữ history trong RAM gọn.

        Lưu ý:
        - Đề yêu cầu tối thiểu 5 câu hỏi gần nhất.
        - Giai đoạn sau sẽ thêm auto-summarization bằng LLM khi vượt token threshold.
        - Ở MVP, mình giữ nhiều hơn một chút để debug: max = window * 4 messages.
        """

        max_messages = max(10, self.settings.session_history_window * 4)
        if len(session.history) > max_messages:
            session.history = session.history[-max_messages:]

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
            expired_ids = []
            for session_id, session in self._sessions.items():
                inactive_seconds = (now - session.last_active_at).total_seconds()
                if inactive_seconds > ttl:
                    expired_ids.append(session_id)

            for session_id in expired_ids:
                self._sessions.pop(session_id, None)
                removed += 1

        return removed


# Singleton tiện dùng cho MVP FastAPI.
session_store = SessionStore()