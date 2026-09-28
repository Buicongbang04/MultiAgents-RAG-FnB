"""
Hội thoại nhiều lượt: nhận diện câu hỏi nối tiếp (follow-up) và dựng câu hỏi độc lập.

Router phân loại từng câu độc lập, nên câu trả lời ngắn cho câu hỏi làm rõ của bot
("Gợi ý món ít ngọt" → bot hỏi "…hay mát lạnh hơn ạ?" → khách: "lạnh") bị coi là
nhiễu, và RAG chỉ tìm với "lạnh" nên mất tiêu chí "ít ngọt".

Resolver chạy ngay sau router:
  1. Nhận diện follow-up: câu ngắn, gần lượt nghiệp vụ trước, và router không thấy
     intent / có từ nối ("còn…", "thì sao", "…hơn") / tinh chỉnh cùng intent.
  2. Kế thừa intent lượt trước.
  3. Dựng standalone query cho RAG + cache key: LLM viết lại (nếu có), lỗi thì ghép
     các câu gần nhất của cùng chủ đề.

Trạng thái lưu trong SessionState.last_intent + metadata["conversation"].
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.core.config import get_lexicon, get_settings
from app.core.constants import Intent
from app.core.logging import get_logger
from app.core.schemas import Message, RouterOutput, SessionState
from app.prompts.conversation import QUERY_REWRITE_SYSTEM_PROMPT

logger = get_logger(__name__)

_STATE_KEY = "conversation"
# NEXT_ACTION của order agent mà bot đang chờ khách trả lời.
PENDING_ORDER_ACTIONS = {"ask_item", "ask_size", "size_unavailable"}


@dataclass
class ResolvedTurn:
    intent: Intent
    query: str                      # standalone query cho RAG + cache
    is_followup: bool
    segments: List[str]             # các câu user của chủ đề hiện tại
    metadata: Dict[str, Any] = field(default_factory=dict)

    def apply_to(self, router_output: RouterOutput) -> RouterOutput:
        metadata = dict(router_output.metadata)
        metadata["conversation"] = self.metadata
        return router_output.model_copy(update={"action": self.intent, "metadata": metadata})


class ConversationResolver:
    def __init__(self) -> None:
        self.config = get_settings().conversation
        lexicon = get_lexicon().conversation
        self.followup_markers = [re.compile(p) for p in lexicon.followup_markers]
        self.greeting_patterns = [
            re.compile(rf"(?<!\w){re.escape(word)}(?!\w)") for word in lexicon.greeting_words
        ]
        self.menu_item_hints = get_lexicon().router_rules.menu_item_hints

    @staticmethod
    def _state(session: SessionState) -> Dict[str, Any]:
        return session.metadata.setdefault(
            _STATE_KEY,
            {"turn": 0, "last_business_turn": None, "segments": []},
        )

    def _is_greeting(self, normalized: str) -> bool:
        return any(p.search(normalized) for p in self.greeting_patterns)

    def _has_followup_marker(self, normalized: str) -> bool:
        return any(p.search(normalized) for p in self.followup_markers)

    @staticmethod
    def _router_is_weak(router_output: RouterOutput) -> bool:
        """Router không có tín hiệu rõ ràng → được phép ghi đè bằng ngữ cảnh."""
        if router_output.action == Intent.IGNORE:
            return True
        metadata = router_output.metadata
        if not str(metadata.get("router_type", "")).startswith("rule"):
            return False
        # "còn trà thì sao?" chỉ khớp tên món trơn → rule router đoán order.
        return set(metadata.get("matched_keywords") or []) <= {"menu_item_alone"}

    def _followup_reason(
        self,
        text: str,
        router_output: RouterOutput,
        session: SessionState,
    ) -> Optional[str]:
        config = self.config
        state = self._state(session)
        last_intent = session.last_intent

        if not config.followup_enabled or last_intent is None:
            return None
        if last_intent.value not in config.followup_intents:
            return None
        if state["last_business_turn"] is None:
            return None
        if state["turn"] - state["last_business_turn"] > config.followup_max_turn_gap:
            return None

        normalized = text.lower().strip()
        if len(normalized.split()) > config.followup_max_words or self._is_greeting(normalized):
            return None

        routed = router_output.action
        if routed == Intent.IGNORE:
            return "router_no_intent"
        if self._has_followup_marker(normalized) and (
            routed == last_intent or self._router_is_weak(router_output)
        ):
            return "followup_marker"
        if routed == last_intent and routed.value in config.refine_intents:
            return "refine_same_intent"
        # Bot vừa hỏi chọn món/size ("cho 2 trà" → "trà nào ạ?") → câu trả lời
        # "trà vải" là bổ sung cho đơn đang dở, không phải đơn mới (giữ số lượng 2).
        if routed == Intent.ORDER == last_intent and state.get("order_pending"):
            return "answer_to_order_question"
        # "size L", "thêm 1 ly nữa": order nhưng không nêu món → sửa món đang đặt.
        if (
            routed == Intent.ORDER == last_intent
            and not any(hint in normalized for hint in self.menu_item_hints)
        ):
            return "order_modifier_without_item"
        return None

    async def resolve(
        self,
        session: SessionState,
        text: str,
        router_output: RouterOutput,
        history: List[Message],
    ) -> ResolvedTurn:
        state = self._state(session)
        state["turn"] += 1

        reason = self._followup_reason(text, router_output, session)
        if reason is None:
            return ResolvedTurn(
                intent=router_output.action,
                query=text,
                is_followup=False,
                segments=[text],
                metadata={"is_followup": False},
            )

        segments = state["segments"] + [text]
        max_segments = self.config.max_query_segments
        if len(segments) > max_segments:
            # Giữ câu mở đầu chủ đề ("Gợi ý món ít ngọt") + các câu gần nhất.
            segments = [segments[0]] + segments[-(max_segments - 1):]
        concatenated = " ".join(segments)
        rewritten = await self._rewrite_with_llm(history, text)
        query = rewritten or concatenated

        metadata = {
            "is_followup": True,
            "reason": reason,
            "inherited_intent": session.last_intent.value,
            "router_intent": router_output.action.value,
            "standalone_query": query,
            "rewrite": "llm" if rewritten else "concat",
        }
        logger.info(
            "Follow-up session=%s reason=%s intent=%s query=%r",
            session.session_id, reason, session.last_intent.value, query,
        )
        return ResolvedTurn(
            intent=session.last_intent,
            query=query,
            is_followup=True,
            segments=segments,
            metadata=metadata,
        )

    def remember(
        self,
        session: SessionState,
        resolved: ResolvedTurn,
        order_action: Optional[str] = None,
    ) -> None:
        """
        Lưu chủ đề hiện tại. Lượt ignore thật (chào hỏi) không xoá ngữ cảnh.
        order_action: NEXT_ACTION của order agent — đang hỏi món/size thì đơn còn dở.
        """
        if resolved.intent.value not in self.config.followup_intents:
            return
        state = self._state(session)
        session.last_intent = resolved.intent
        state["last_business_turn"] = state["turn"]
        state["segments"] = resolved.segments
        state["order_pending"] = order_action in PENDING_ORDER_ACTIONS

    def mark_dialogue_turn(self, session: SessionState, topic: str) -> None:
        """
        Lượt do order dialogue manager xử lý ("ok", "xem đơn"…).
          close → món đang bàn đã xong, câu sau không ghép với món cũ
          keep  → bot hỏi lại về chính món đó ("đổi size hay số lượng?") → giữ ngữ cảnh
          none  → câu đáp xã giao, không động tới ngữ cảnh
        """
        state = self._state(session)
        state["turn"] += 1
        if topic == "none":
            return
        session.last_intent = Intent.ORDER
        state["last_business_turn"] = state["turn"]
        state["order_pending"] = topic == "keep"
        if topic == "close":
            state["segments"] = []

    async def _rewrite_with_llm(self, history: List[Message], text: str) -> Optional[str]:
        settings = get_settings()
        if not self.config.rewrite_with_llm or settings.llm.backend == "mock" or not history:
            return None

        from app.llm import get_llm_client
        from app.llm.base import LLMGenerateRequest

        speaker = {"user": "Khách", "assistant": "Bot"}
        transcript = "\n".join(
            f"{speaker.get(m.role, m.role)}: {m.content}" for m in history if m.role in speaker
        )
        try:
            response = await get_llm_client().generate(
                LLMGenerateRequest(
                    system_prompt=QUERY_REWRITE_SYSTEM_PROMPT,
                    user_prompt=f"HỘI THOẠI:\n{transcript}\n\nCÂU CUỐI CỦA KHÁCH: {text}",
                    temperature=self.config.rewrite_temperature,
                    max_tokens=self.config.rewrite_max_tokens,
                )
            )
        except Exception as exc:
            logger.warning("Query rewrite failed: %s", exc)
            return None

        if response.metadata.get("fallback_used"):
            return None
        rewritten = response.text.strip().strip('"“”').splitlines()[0].strip() if response.text.strip() else ""
        return rewritten or None


conversation_resolver = ConversationResolver()
