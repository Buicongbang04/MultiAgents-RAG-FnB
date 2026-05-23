import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Tuple

from app.core.constants import Intent, Language

@dataclass(frozen=True)
class IntentMatch:
    intent: Intent
    score: float
    matched_keywords: List[str]
    reason: str

def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    return text


def detect_language(text: str) -> Language:
    normalized = normalize_text(text)

    vietnamese_markers = [
        "anh", "chị", "em", "cho", "món", "quán", "cà phê", "trà",
        "sữa", "đá", "nóng", "ngon", "giá", "wifi", "mấy giờ",
    ]
    english_markers = [
        "hello", "hi", "order", "coffee", "tea", "recommend", "wifi",
        "open", "close", "price", "menu", "pay",
    ]

    vi_hits = sum(1 for marker in vietnamese_markers if marker in normalized)
    en_hits = sum(1 for marker in english_markers if marker in normalized)

    if vi_hits > en_hits:
        return Language.VI
    if en_hits > vi_hits:
        return Language.EN
    return Language.UNKNOWN


ORDER_KEYWORDS = [
    "cho anh", "cho chị", "cho em", "cho tôi", "cho mình",
    "lấy anh", "lấy chị", "lấy em", "lấy tôi", "lấy mình",
    "gọi", "đặt", "order", "thêm", "bớt", "bỏ",
    "tính tiền", "thanh toán", "mua",
    "một ly", "1 ly", "hai ly", "2 ly",
    "size s", "size m", "size l",
    "i want", "i'd like", "can i get", "give me",
    "add", "remove", "checkout", "pay", "one cup", "two cups",
]

CONSULTANT_KEYWORDS = [
    "gợi ý", "tư vấn", "có gì ngon", "món nào ngon", "ngon không",
    "nên uống", "nên chọn", "ít ngọt", "ngọt ít", "ít béo",
    "không đắng", "trời nóng", "trời mưa", "rẻ không", "dưới",
    "phù hợp", "recommend", "suggest", "what is good", "what's good",
    "less sweet", "not too sweet", "cheap", "budget", "best drink",
    "what should i drink",
]

FAQ_KEYWORDS = [
    "wifi", "mật khẩu", "password", "mấy giờ", "giờ mở cửa",
    "giờ đóng cửa", "đóng cửa", "mở cửa", "địa chỉ", "ở đâu",
    "nhà vệ sinh", "toilet", "gửi xe", "giữ xe", "thanh toán momo",
    "momo", "visa", "chuyển khoản", "vat", "hóa đơn", "xuất hóa đơn",
    "khuyến mãi", "voucher", "giao hàng", "delivery", "chính sách",
    "opening hours", "close time", "open time", "where is", "address",
    "parking", "invoice", "promotion", "policy",
]

IGNORE_KEYWORDS = [
    "ừm", "ờ", "alo", "hello", "haha", "hehe", "test", "noise", "...",
]

QUESTION_MARKERS = [
    "?", "không", "ko", "hông", "à", "hả", "chưa",
    "what", "which", "where", "when", "how",
]

MENU_ITEM_HINTS = [
    "cà phê", "cafe", "coffee", "bạc xỉu", "latte", "trà", "tea",
    "trà sen", "trà đào", "freeze", "sinh tố", "bánh",
]

def _keyword_score(text: str, keywords: List[str], base_score: float) -> Tuple[float, List[str]]:
    matched = [kw for kw in keywords if kw in text]
    if not matched:
        return 0.0, []

    score = base_score + min(0.25, 0.05 * (len(matched) - 1))
    return score, matched

def classify_by_rules(text: str) -> IntentMatch:
    normalized = normalize_text(text)

    order_score, order_matches = _keyword_score(normalized, ORDER_KEYWORDS, 0.70)
    consultant_score, consultant_matches = _keyword_score(normalized, CONSULTANT_KEYWORDS, 0.65)
    faq_score, faq_matches = _keyword_score(normalized, FAQ_KEYWORDS, 0.75)
    ignore_score, ignore_matches = _keyword_score(normalized, IGNORE_KEYWORDS, 0.45)

    has_menu_hint = any(hint in normalized for hint in MENU_ITEM_HINTS)
    has_question_marker = any(marker in normalized for marker in QUESTION_MARKERS)

    # Có số lượng + món → order.
    quantity_pattern = r"\b(1|2|3|4|5|một|hai|ba|bốn|năm)\b"
    if has_menu_hint and re.search(quantity_pattern, normalized):
        order_score += 0.20
        order_matches.append("quantity+menu_item")

    # Tên món đơn thuần (không có số lượng, không có FAQ/consultant signal) → order.
    if has_menu_hint and not faq_matches and not consultant_matches:
        order_score = max(order_score, 0.55)
        if "menu_item_alone" not in order_matches:
            order_matches.append("menu_item_alone")

    # Hỏi khẩu vị/gợi ý/món ngon → consultant.
    preference_words = ["ngon", "ít ngọt", "rẻ", "recommend", "suggest", "good"]
    if has_question_marker and has_menu_hint and any(word in normalized for word in preference_words):
        consultant_score += 0.15
        consultant_matches.append("question_about_menu_preference")

    # Greeting/noise rất ngắn → ignore.
    if len(normalized) <= 8 and ignore_matches:
        ignore_score += 0.25

    candidates: Dict[Intent, Tuple[float, List[str], str]] = {
        Intent.ORDER: (order_score, order_matches, "matched order/action keywords"),
        Intent.CONSULTANT: (consultant_score, consultant_matches, "matched recommendation/preference keywords"),
        Intent.FAQ: (faq_score, faq_matches, "matched FAQ/store-info keywords"),
        Intent.IGNORE: (ignore_score, ignore_matches, "matched greeting/noise keywords"),
    }

    # Rule đã chốt: nếu có tín hiệu order rõ ràng thì ưu tiên order.
    if order_score >= 0.70 and order_matches:
        return IntentMatch(
            intent=Intent.ORDER,
            score=min(order_score, 0.99),
            matched_keywords=order_matches,
            reason="order priority: explicit ordering/payment/modification signal",
        )

    best_intent = max(candidates.keys(), key=lambda intent: candidates[intent][0])
    best_score, best_matches, reason = candidates[best_intent]

    if best_score <= 0.0 or not best_matches:
        return IntentMatch(
            intent=Intent.IGNORE,
            score=0.50,
            matched_keywords=[],
            reason="no reliable business intent matched",
        )

    return IntentMatch(
        intent=best_intent,
        score=min(best_score, 0.99),
        matched_keywords=best_matches,
        reason=reason,
    )