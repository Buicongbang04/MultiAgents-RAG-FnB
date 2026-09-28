"""
Dialogue state tracking cho order: giỏ hàng + điều bot đang chờ khách trả lời.

Router phân loại từng câu độc lập nên "ok" sau câu "Anh/chị xác nhận giúp em nhé?"
bị coi là nhiễu. Tầng này chạy TRƯỚC router và xử lý tất định (không LLM/RAG):

  pending = confirm_line      "ok" → thêm vào giỏ, hỏi "cần thêm gì không?"
                              "không" → bỏ món, hỏi muốn đổi gì
  pending = anything_else     "không"/"hết rồi" → đọc lại đơn + tổng, xin chốt
                              "có" → hỏi món thêm
  pending = confirm_checkout  "ok" → chốt đơn · "không" → hỏi muốn chỉnh gì
  pending = order_question    "thôi" → bỏ câu hỏi món/size đang dở
  mọi lúc                     "xem đơn", "bỏ trà vải", "chốt đơn", "huỷ đơn"

Chỉ khớp TRỌN câu (sau khi bỏ trợ từ); câu khác đi tiếp pipeline bình thường
(vd. "size M thôi" → follow-up resolver sửa món đang chờ xác nhận).
Tổng tiền do code tính — LLM không bao giờ tự cộng tiền.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.agents.context_builders import format_vnd
from app.core.config import get_lexicon, get_settings
from app.core.constants import Intent
from app.core.schemas import SessionState

_STATE_KEY = "order"
ORDER_QUESTION_ACTIONS = {"ask_item", "ask_size", "size_unavailable"}


class Topic:
    """Lượt này ảnh hưởng thế nào tới ngữ cảnh follow-up của ConversationResolver."""
    CLOSE = "close"   # món đang bàn đã xong (đã thêm/bỏ/chốt) → câu sau là món mới
    KEEP = "keep"     # bot hỏi lại về chính món đó ("đổi size hay số lượng?")
    NONE = "none"     # câu đáp xã giao, không động tới ngữ cảnh


@dataclass
class DialogueTurn:
    answer: str
    action: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    topic: str = Topic.CLOSE


def _line_label(line: dict) -> str:
    size = f" size {line['size']}" if line.get("size") else ""
    return f"{line['quantity']} {line['item']}{size}"


def _line_total(line: dict) -> int:
    return line["quantity"] * line["unit_price"]


class OrderDialogueManager:
    def __init__(self) -> None:
        self.config = get_settings().order_dialogue
        lexicon = get_lexicon().order_dialogue
        self.particles = set(lexicon.particles)
        self.affirm = set(lexicon.affirm)
        self.negate = set(lexicon.negate)
        self.checkout = set(lexicon.checkout)
        self.view_cart = set(lexicon.view_cart)
        self.cancel_order = set(lexicon.cancel_order)
        self.remove_pattern = re.compile(lexicon.remove_pattern)
        self.generic_item_words = set(get_lexicon().order.generic_item_words)
        # "bỏ đá", "bỏ đường" là tuỳ chỉnh, không phải bỏ món.
        self.non_item_words = set(get_lexicon().order.non_item_words)

    # ── state ────────────────────────────────────────────────────────────────
    @staticmethod
    def state(session: SessionState) -> Dict[str, Any]:
        return session.metadata.setdefault(
            _STATE_KEY, {"cart": [], "pending": None, "submitted": []}
        )

    def _normalize(self, text: str) -> str:
        text = re.sub(r"[^\w\s']", " ", text.lower())
        return re.sub(r"\s+", " ", text).strip()

    def _phrase(self, text: str) -> str:
        """Bỏ trợ từ: "ok em nhé" → "ok", "đúng rồi đó ạ" → "đúng rồi"."""
        tokens = [t for t in self._normalize(text).split() if t not in self.particles]
        return " ".join(tokens)

    # ── rendering ────────────────────────────────────────────────────────────
    def _cart_summary(self, cart: List[dict], total_label: str = "Tạm tính") -> str:
        lines = [
            f"• {_line_label(line)} — {format_vnd(_line_total(line))}"
            for line in cart
        ]
        total = sum(_line_total(line) for line in cart)
        return "\n".join(lines) + f"\n**{total_label}: {format_vnd(total)}**"

    def _cart_metadata(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "cart": state["cart"],
            "total": sum(_line_total(line) for line in state["cart"]),
            "pending": (state["pending"] or {}).get("type"),
        }

    def _turn(
        self,
        state: Dict[str, Any],
        answer: str,
        action: str,
        topic: str = Topic.CLOSE,
    ) -> DialogueTurn:
        return DialogueTurn(
            answer=answer, action=action, metadata=self._cart_metadata(state), topic=topic
        )

    # ── actions ──────────────────────────────────────────────────────────────
    def _add_line(self, state: Dict[str, Any], line: dict) -> None:
        for existing in state["cart"]:
            if existing["variant_id"] == line["variant_id"]:
                existing["quantity"] += line["quantity"]
                return
        state["cart"].append(dict(line))

    def _target_tokens(self, target: str) -> set:
        return set(re.findall(r"\w+", target)) - self.generic_item_words - self.non_item_words

    def _find_cart_line(self, cart: List[dict], target: str) -> Optional[dict]:
        """"bỏ trà bưởi" → dòng "Trà bưởi mật ong": từ khách nói nằm trong tên món, duy nhất."""
        for line in cart:
            if line["item"].lower() in target:
                return line
        target_tokens = self._target_tokens(target)
        if not target_tokens:
            return None
        matches = [
            line for line in cart
            if target_tokens <= set(re.findall(r"\w+", line["item"].lower()))
        ]
        return matches[0] if len(matches) == 1 else None

    def _checkout(self, state: Dict[str, Any]) -> DialogueTurn:
        state["pending"] = {"type": "confirm_checkout"}
        return self._turn(
            state,
            "Dạ, em đọc lại đơn của mình ạ:\n"
            f"{self._cart_summary(state['cart'], total_label='Tổng cộng')}\n\n"
            "Anh/chị xác nhận chốt đơn này giúp em nhé?",
            "checkout_summary",
        )

    def _anything_else(self, state: Dict[str, Any], prefix: str) -> DialogueTurn:
        state["pending"] = {"type": "anything_else"}
        return self._turn(
            state,
            f"{prefix}\n\nĐơn hiện tại:\n{self._cart_summary(state['cart'])}\n\n"
            "Anh/chị cần gọi thêm gì nữa không ạ?",
            "cart_updated",
        )

    # ── entry point ──────────────────────────────────────────────────────────
    def handle(self, session: SessionState, text: str) -> Optional[DialogueTurn]:
        if not self.config.enabled:
            return None

        state = self.state(session)
        cart: List[dict] = state["cart"]
        pending: Optional[dict] = state["pending"]
        normalized = self._normalize(text)
        phrase = self._phrase(text)

        # Lệnh giỏ hàng — dùng được bất cứ lúc nào.
        if phrase in self.cancel_order and (cart or pending):
            state["cart"], state["pending"] = [], None
            return self._turn(
                state,
                "Dạ, em đã huỷ đơn ạ. Anh/chị cần em hỗ trợ gì thêm không ạ?",
                "order_cancelled",
            )

        if phrase in self.view_cart:
            if not cart:
                return self._turn(
                    state,
                    "Dạ, đơn của mình hiện chưa có món nào ạ. Anh/chị muốn gọi món gì ạ?",
                    "view_empty_cart",
                )
            return self._anything_else(state, "Dạ, đơn của mình đang có:")

        removal = self.remove_pattern.match(normalized)
        if removal and not self._target_tokens(removal.group(2)):
            # "bỏ đá", "bớt đường": tuỳ chỉnh món — chưa hỗ trợ, nói thật thay vì đoán.
            # Giữ nguyên pending để "ok"/"không" sau đó vẫn đúng ngữ cảnh.
            return self._turn(
                state,
                "Dạ, hiện em chưa ghi được tuỳ chỉnh (ít đá, không đường…) qua chat ạ. "
                "Anh/chị nhắn giúp nhân viên tại quầy khi nhận món nhé!",
                "customization_unsupported",
                topic=Topic.NONE,
            )
        if removal:
            target = removal.group(2)
            if not cart:
                return self._turn(
                    state,
                    "Dạ, đơn của mình hiện chưa có món nào ạ. Anh/chị muốn gọi món gì ạ?",
                    "remove_from_empty_cart",
                )
            line = self._find_cart_line(cart, target)
            if line is None:
                return self._anything_else(
                    state, f"Dạ, trong đơn chưa có món \"{target}\" ạ."
                )
            if line:
                cart.remove(line)
                if not cart:
                    state["pending"] = None
                    return self._turn(
                        state,
                        f"Dạ, em đã bỏ **{_line_label(line)}** ạ. Đơn hiện chưa có món nào, "
                        "anh/chị muốn gọi món gì ạ?",
                        "line_removed",
                    )
                return self._anything_else(state, f"Dạ, em đã bỏ **{_line_label(line)}** ạ.")

        if phrase in self.checkout:
            if (pending or {}).get("type") == "confirm_line":
                self._add_line(state, pending["line"])  # "chốt đơn" = đồng ý luôn món đang chờ
            if state["cart"]:
                return self._checkout(state)
            if pending is None:
                return self._turn(
                    state,
                    "Dạ, đơn của mình hiện chưa có món nào ạ. Anh/chị muốn gọi món gì ạ?",
                    "checkout_empty_cart",
                )

        if pending is None:
            # "ok"/"không" không gắn với câu hỏi nào (vd. sau câu trả lời FAQ) → đáp xã giao
            # thay vì "Dạ em nghe ạ, anh/chị muốn đặt món…".
            if phrase in self.affirm or phrase in self.negate:
                return self._turn(
                    state,
                    "Dạ vâng ạ! Anh/chị cần em hỗ trợ gì thêm không ạ? "
                    "(đặt món, gợi ý món, xem đơn…)",
                    "acknowledge",
                    topic=Topic.NONE,
                )
            return None

        kind = pending["type"]
        is_affirm = phrase in self.affirm
        is_negate = phrase in self.negate or phrase in self.checkout

        if kind == "confirm_line":
            if is_affirm:
                self._add_line(state, pending["line"])
                return self._anything_else(
                    state, f"Dạ, em đã thêm **{_line_label(pending['line'])}** vào đơn ạ."
                )
            if phrase in self.negate:
                state["pending"] = None
                return self._turn(
                    state,
                    "Dạ vâng, em chưa thêm món này ạ. Anh/chị muốn đổi size, số lượng "
                    "hay chọn món khác ạ?",
                    "line_rejected",
                    topic=Topic.KEEP,
                )

        elif kind == "anything_else":
            if is_negate:
                return self._checkout(state)
            if is_affirm:
                state["pending"] = None
                return self._turn(state, "Dạ, anh/chị muốn gọi thêm món gì ạ?", "ask_more")

        elif kind == "confirm_checkout":
            if is_affirm:
                summary = self._cart_summary(cart, total_label="Tổng cộng")
                state["submitted"].append(list(cart))
                state["cart"], state["pending"] = [], None
                return self._turn(
                    state,
                    "Dạ, em đã chốt đơn ạ 🎉\n"
                    f"{summary}\n\n"
                    "Anh/chị vui lòng thanh toán tại quầy, đồ uống sẽ có trong ít phút. "
                    "Cảm ơn anh/chị ạ!",
                    "order_submitted",
                )
            if phrase in self.negate:
                state["pending"] = None
                return self._turn(
                    state,
                    "Dạ, anh/chị muốn chỉnh gì trong đơn ạ? (đổi size, bớt món, thêm món…)",
                    "checkout_rejected",
                )

        elif kind == "order_question" and phrase in self.negate:
            state["pending"] = None
            return self._turn(
                state, "Dạ vâng ạ. Anh/chị cần em hỗ trợ gì thêm không ạ?", "question_dropped"
            )

        return None

    def pending_reminder(self, session: SessionState, intent: Intent) -> Optional[str]:
        """
        Khách hỏi FAQ/tư vấn khi còn món chờ xác nhận → trả lời xong nhắc lại món,
        như nhân viên thật ("…À, món 1 Latte size L anh/chị xác nhận giúp em nhé?").
        """
        if intent in (Intent.ORDER, Intent.IGNORE):
            return None
        if self.config.pending_on_other_intents != "remind":
            return None
        pending = self.state(session)["pending"]
        if (pending or {}).get("type") != "confirm_line":
            return None
        return (
            f"À, món **{_line_label(pending['line'])}** lúc nãy "
            "anh/chị xác nhận giúp em để em thêm vào đơn nhé?"
        )

    def after_agent(
        self,
        session: SessionState,
        intent: Intent,
        order_metadata: Optional[dict],
    ) -> None:
        """Cập nhật điều bot đang chờ sau một lượt đi qua agent."""
        state = self.state(session)

        if intent == Intent.ORDER:
            order_metadata = order_metadata or {}
            action = order_metadata.get("action")
            if action == "confirm" and order_metadata.get("line"):
                state["pending"] = {"type": "confirm_line", "line": order_metadata["line"]}
            elif action in ORDER_QUESTION_ACTIONS:
                state["pending"] = {"type": "order_question", "action": action}
            else:
                state["pending"] = None
            return

        # Hỏi FAQ/tư vấn giữa chừng: "remind" giữ món chờ (đã nhắc lại trong câu trả lời),
        # "clear" bỏ luôn. Lượt ignore (chào hỏi) luôn giữ nguyên trạng thái.
        if intent != Intent.IGNORE and self.config.pending_on_other_intents == "clear":
            state["pending"] = None


order_dialogue_manager = OrderDialogueManager()
