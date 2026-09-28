"""
Slot filling cho order: món → size → số lượng → bước tiếp theo.

Một nhân viên order không xác nhận "Trà bưởi mật ong size S" khi khách chưa chọn
size. Module này quyết định NEXT_ACTION một cách tất định; cùng kết quả được dùng
để dựng context cho LLM và câu trả lời dự phòng (mock / LLM lỗi) → hai đường nhất quán.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from app.core.config import get_lexicon
from app.core.constants import SourceType
from app.rag.menu import get_menu_names, get_menu_variants, get_menu_vocabulary


class OrderAction(str, Enum):
    NO_MATCH = "no_match"                # không tìm thấy món
    ASK_ITEM = "ask_item"                # nhiều món khớp mơ hồ → hỏi chọn món
    ASK_SIZE = "ask_size"                # món có nhiều size, khách chưa chọn
    SIZE_UNAVAILABLE = "size_unavailable"
    CONFIRM = "confirm"                  # đủ thông tin → xin khách xác nhận


@dataclass
class OrderResolution:
    action: OrderAction
    item_name: Optional[str] = None
    category: Optional[str] = None
    variants: List[dict] = field(default_factory=list)   # mọi size của món
    size: Optional[str] = None                            # size khách yêu cầu
    quantity: int = 1
    selected: Optional[dict] = None                       # variant đã chốt
    candidates: List[str] = field(default_factory=list)  # khi ASK_ITEM
    unknown_words: List[str] = field(default_factory=list)  # khi NO_MATCH vì món ngoài menu
    candidate_variants: dict = field(default_factory=dict)  # khi ASK_ITEM: size/giá từng món
    availability_question: bool = False                   # "có bán X không?"
    requested_phrase: str = ""                             # "phở bò" — món khách nhắc tới

    def selected_line(self) -> Optional[dict]:
        """Dòng đơn đang chờ khách xác nhận (chỉ có khi action = CONFIRM)."""
        if self.action != OrderAction.CONFIRM or not self.selected:
            return None
        return {
            "item": self.item_name,
            "size": self.selected["size"] if len(self.variants) > 1 else None,
            "quantity": self.quantity,
            "unit_price": self.selected["price"],
            "variant_id": self.selected["id"],
        }

    def to_metadata(self) -> dict:
        return {
            "line": self.selected_line(),
            "action": self.action.value,
            "item": self.item_name,
            "size": self.size,
            "quantity": self.quantity,
            "available_sizes": [v["size"] for v in self.variants],
            "candidates": self.candidates,
            "unknown_words": self.unknown_words,
            "availability_question": self.availability_question,
        }


def item_name(source) -> str:
    return source.text.split("|")[0].strip()


def distinct_menu_items(sources) -> List[str]:
    """Tên món khác nhau theo thứ tự xếp hạng (mỗi size là một nguồn riêng)."""
    names: List[str] = []
    for source in sources:
        if source.source_type != SourceType.MENU:
            continue
        name = item_name(source)
        if name and name not in names:
            names.append(name)
    return names


def extract_size(text: str) -> Optional[str]:
    lexicon = get_lexicon().order
    lowered = text.lower()

    # Lần nhắc sau cùng thắng: "size S… à thôi size L" → L.
    best: Optional[tuple[int, str]] = None
    for size, patterns in lexicon.size_patterns.items():
        for pattern in patterns:
            for match in re.finditer(pattern, lowered):
                if best is None or match.start() > best[0]:
                    best = (match.start(), size)
    for match in re.finditer(lexicon.generic_size_pattern, lowered):
        if best is None or match.start() > best[0]:
            best = (match.start(), match.group(1).upper())
    if best:
        return best[1]

    # Trả lời cộc lốc cho câu hỏi size ("M", "vừa nhé") nằm ở cuối standalone query.
    tokens = re.findall(r"\w+", lowered)
    while tokens and tokens[-1] in lexicon.trailing_particles:
        tokens.pop()
    if tokens:
        for size, words in lexicon.bare_size_words.items():
            if tokens[-1] in words:
                return size
    return None


def extract_quantity(text: str) -> int:
    lexicon = get_lexicon().order
    lowered = text.lower()

    found: List[tuple[int, int]] = []
    for match in re.finditer(lexicon.quantity_digit_pattern, lowered):
        found.append((match.start(), int(match.group(1))))
    for match in re.finditer(lexicon.quantity_word_pattern, lowered):
        found.append((match.start(), lexicon.quantity_words[match.group(1)]))

    if not found:
        return 1
    quantity = max(found)[1]  # lần nhắc sau cùng
    return quantity if quantity > 0 else 1


def _pick_item(text: str, names: List[str]) -> Optional[str]:
    lowered = text.lower()

    # Tên món nằm trọn trong câu → chọn tên dài nhất ("trà đào cam sả" > "trà").
    contained = [name for name in names if name.lower() in lowered]
    if contained:
        return max(contained, key=len)

    # Khớp một phần theo từ đặc trưng: "trà bưởi" → "Trà bưởi mật ong".
    # "cho 1 trà" chỉ khớp từ chung → không đoán; hoà điểm → mơ hồ → hỏi lại.
    generic = set(get_lexicon().order.generic_item_words)
    tokens = set(re.findall(r"\w+", lowered)) - generic
    scored = []
    for name in names:
        distinctive = set(re.findall(r"\w+", name.lower())) - generic
        scored.append((len(distinctive & tokens), name))
    scored.sort(key=lambda x: x[0], reverse=True)

    if scored and scored[0][0] > 0:
        if len(scored) == 1 or scored[1][0] < scored[0][0]:
            return scored[0][1]
    return None


def unknown_item_words(text: str, names: List[str]) -> List[str]:
    """
    Từ khách nhắc mà menu không hề có ("phở", "pizza") → khách gọi món ngoài menu.

    Guardrail ở tầng món, không dựa vào ngưỡng reranker: "phở bò" vẫn khớp chữ "bò"
    của "Sừng Bò" đủ để vượt ngưỡng, nhưng "phở" không có trong từ vựng menu.
    Bỏ qua khi tên một món nằm trọn trong câu.
    """
    lowered = text.lower()
    if any(name.lower() in lowered for name in names):
        return []

    lexicon = get_lexicon()
    ignored = (
        set(lexicon.retrieval.stopwords)
        | set(lexicon.order.non_item_words)
        | set(lexicon.order.trailing_particles)
        | {w for words in lexicon.order.bare_size_words.values() for w in words}
    )
    vocabulary = get_menu_vocabulary()
    return [
        token for token in re.findall(r"\w+", lowered)
        if token not in ignored and not token.isdigit() and token not in vocabulary
    ]


def _item_tokens(text: str) -> set[str]:
    """Từ trong câu có thể là một phần tên món (bỏ stopword, trợ từ, size, số lượng)."""
    lexicon = get_lexicon()
    ignored = (
        set(lexicon.retrieval.stopwords)
        | set(lexicon.order.trailing_particles)
        | {w for words in lexicon.order.bare_size_words.values() for w in words}
        | set(lexicon.order.quantity_words)
        | {"size", "cỡ", "ly", "cốc", "phần", "cái", "chiếc", "suất"}
    )
    return {t for t in re.findall(r"\w+", text.lower()) if t not in ignored and not t.isdigit()}


def _menu_candidates(text: str, all_names: List[str]) -> List[str]:
    """
    Món trong TOÀN BỘ menu có tên chứa đủ các từ khách nói ("cho 2 trà" → mọi món có
    chữ "trà"; "cà phê sữa không đường" → Cà phê sữa đá / nóng). Từ không thuộc tên
    món nào ("không", "đường") là tuỳ chỉnh, bỏ qua khi lọc.
    """
    name_tokens = {name: set(re.findall(r"\w+", name.lower())) for name in all_names}
    name_vocabulary = set().union(*name_tokens.values()) if name_tokens else set()
    tokens = _item_tokens(text) & name_vocabulary
    if not tokens:
        return []
    return [name for name in all_names if tokens <= name_tokens[name]]


def _is_availability_question(text: str) -> bool:
    from app.agents.intent_rules import is_availability_question

    lowered = text.lower()
    has_menu_hint = any(hint in lowered for hint in get_lexicon().router_rules.menu_item_hints)
    return is_availability_question(lowered, has_menu_hint)


def resolve_order(text: str, sources, latest_text: Optional[str] = None) -> OrderResolution:
    """
    text: standalone query (đã ghép follow-up: "trà vải" + "L").
    latest_text: câu mới nhất của khách — quyết định câu hỏi còn món ("có bán trà không")
    chỉ cho đúng lượt đó, không lặp "Dạ có ạ!" ở các lượt trả lời size sau.
    """
    quantity = extract_quantity(text)
    size = extract_size(text)
    availability = _is_availability_question(latest_text or text)
    all_names = list(get_menu_names())
    lowered = text.lower()

    def result(action: OrderAction, **kwargs) -> OrderResolution:
        return OrderResolution(
            action=action,
            quantity=quantity,
            size=size,
            availability_question=availability,
            **kwargs,
        )

    # Guardrail món ngoài menu ("phở", "pizza") — trước mọi bước đoán món.
    unknown = unknown_item_words(text, all_names)
    if unknown:
        item_tokens = _item_tokens(text)
        phrase = " ".join(t for t in re.findall(r"\w+", lowered) if t in item_tokens)
        return result(OrderAction.NO_MATCH, unknown_words=unknown, requested_phrase=phrase)

    # 1) Tên món nằm trọn trong câu → tra thẳng menu, không phụ thuộc top-k của RAG
    #    (mỗi size là một node nên top-5 chỉ chứa ~2 món; "latte nóng" bị reranker
    #    đẩy Latte xuống dưới "Cà phê sữa nóng").
    exact = [name for name in all_names if name.lower() in lowered]
    chosen = max(exact, key=len) if exact else None

    # 2) Từ đặc trưng duy nhất trên toàn menu: "trà bưởi" → Trà bưởi mật ong.
    if chosen is None:
        chosen = _pick_item(text, all_names)

    # 3) Mơ hồ → liệt kê từ menu trong graph (không phải top-k RAG).
    if chosen is None:
        candidates = _menu_candidates(text, all_names) or distinct_menu_items(sources)
        if len(candidates) == 1:
            chosen = candidates[0]
        elif candidates:
            candidates = candidates[: get_lexicon().order.max_item_choices]
            return result(
                OrderAction.ASK_ITEM,
                candidates=candidates,
                candidate_variants=get_menu_variants(candidates),
            )
        else:
            return result(OrderAction.NO_MATCH)

    variants = get_menu_variants([chosen]).get(chosen) or []
    if not variants:
        return result(OrderAction.NO_MATCH)

    resolution = result(
        OrderAction.CONFIRM,
        item_name=chosen,
        category=variants[0]["category"],
        variants=variants,
    )

    if len(variants) == 1:
        # Món một size (bánh…) → không hỏi size.
        resolution.selected = variants[0]
        return resolution

    if size is None:
        resolution.action = OrderAction.ASK_SIZE
        return resolution

    matched = [v for v in variants if v["size"] == size]
    if not matched:
        resolution.action = OrderAction.SIZE_UNAVAILABLE
        return resolution

    resolution.selected = matched[0]
    return resolution
