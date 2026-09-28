import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Tuple

from app.core.config import get_lexicon, get_settings
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
    markers = get_lexicon().language.rule_router_markers

    vi_hits = sum(1 for marker in markers["vi"] if marker in normalized)
    en_hits = sum(1 for marker in markers["en"] if marker in normalized)

    if vi_hits > en_hits:
        return Language.VI
    if en_hits > vi_hits:
        return Language.EN
    return Language.UNKNOWN


def _keyword_score(
    text: str,
    keywords: List[str],
    base_score: float,
    extra_match_bonus: float,
    max_extra_bonus: float,
) -> Tuple[float, List[str]]:
    matched = [kw for kw in keywords if kw in text]
    if not matched:
        return 0.0, []

    score = base_score + min(max_extra_bonus, extra_match_bonus * (len(matched) - 1))
    return score, matched

def is_availability_question(text: str, has_menu_hint: bool) -> bool:
    """
    "có bán phở bò không", "có món cơm gà không" → luôn là hỏi còn món.
    "có trà đào không", "còn bạc xỉu không ạ" → chỉ khi câu có tên/loại món
    (tránh bắt nhầm "có wifi không", "có chỗ ngồi không").
    """
    order_lexicon = get_lexicon().order
    if any(re.search(pattern, text) for pattern in order_lexicon.availability_patterns):
        return True
    return has_menu_hint and bool(re.search(order_lexicon.generic_availability_pattern, text))


def classify_by_rules(text: str) -> IntentMatch:
    normalized = normalize_text(text)
    lexicon = get_lexicon().router_rules
    rules = get_settings().router.rules

    def score(intent: Intent) -> Tuple[float, List[str]]:
        return _keyword_score(
            normalized,
            lexicon.keywords[intent.value],
            rules.base_scores[intent.value],
            rules.extra_match_bonus,
            rules.max_extra_bonus,
        )

    order_score, order_matches = score(Intent.ORDER)
    consultant_score, consultant_matches = score(Intent.CONSULTANT)
    faq_score, faq_matches = score(Intent.FAQ)
    ignore_score, ignore_matches = score(Intent.IGNORE)

    has_menu_hint = any(hint in normalized for hint in lexicon.menu_item_hints)
    has_question_marker = any(marker in normalized for marker in lexicon.question_markers)

    # Hỏi còn món không → order (món ngoài menu sẽ bị guardrail ở order agent chặn).
    # Nhường cho FAQ/tư vấn nếu câu có tín hiệu của chúng ("có cà phê nào ngon không").
    if (
        not faq_matches
        and not consultant_matches
        and is_availability_question(normalized, has_menu_hint)
    ):
        return IntentMatch(
            intent=Intent.ORDER,
            score=rules.order_priority_threshold,
            matched_keywords=["availability_question"],
            reason="availability question: customer asks whether an item is sold",
        )

    # Có số lượng + món → order.
    if has_menu_hint and re.search(lexicon.quantity_pattern, normalized):
        order_score += rules.quantity_menu_bonus
        order_matches.append("quantity+menu_item")

    # Tên món đơn thuần (không có số lượng, không có FAQ/consultant signal) → order.
    if has_menu_hint and not faq_matches and not consultant_matches:
        order_score = max(order_score, rules.menu_item_alone_min_score)
        if "menu_item_alone" not in order_matches:
            order_matches.append("menu_item_alone")

    # Hỏi khẩu vị/gợi ý/món ngon → consultant.
    if (
        has_question_marker
        and has_menu_hint
        and any(word in normalized for word in lexicon.preference_words)
    ):
        consultant_score += rules.preference_question_bonus
        consultant_matches.append("question_about_menu_preference")

    # Greeting/noise rất ngắn → ignore.
    if len(normalized) <= rules.short_noise_max_chars and ignore_matches:
        ignore_score += rules.short_noise_bonus

    candidates: Dict[Intent, Tuple[float, List[str], str]] = {
        Intent.ORDER: (order_score, order_matches, "matched order/action keywords"),
        Intent.CONSULTANT: (consultant_score, consultant_matches, "matched recommendation/preference keywords"),
        Intent.FAQ: (faq_score, faq_matches, "matched FAQ/store-info keywords"),
        Intent.IGNORE: (ignore_score, ignore_matches, "matched greeting/noise keywords"),
    }

    # Câu hỏi thông tin quán ("Có thanh toán momo không?") chứa từ khóa order
    # ("thanh toán") nhưng không có số lượng món → không được ép sang order.
    is_store_info_question = (
        bool(faq_matches)
        and has_question_marker
        and "quantity+menu_item" not in order_matches
    )

    # Rule đã chốt: nếu có tín hiệu order rõ ràng thì ưu tiên order.
    if (
        order_score >= rules.order_priority_threshold
        and order_matches
        and not is_store_info_question
    ):
        return IntentMatch(
            intent=Intent.ORDER,
            score=min(order_score, rules.max_confidence),
            matched_keywords=order_matches,
            reason="order priority: explicit ordering/payment/modification signal",
        )

    best_intent = max(candidates.keys(), key=lambda intent: candidates[intent][0])
    best_score, best_matches, reason = candidates[best_intent]

    if best_score <= 0.0 or not best_matches:
        return IntentMatch(
            intent=Intent.IGNORE,
            score=rules.no_match_score,
            matched_keywords=[],
            reason="no reliable business intent matched",
        )

    return IntentMatch(
        intent=best_intent,
        score=min(best_score, rules.max_confidence),
        matched_keywords=best_matches,
        reason=reason,
    )
