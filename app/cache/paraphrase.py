from __future__ import annotations

from typing import Any, Dict, Optional

from app.core.constants import Intent


def _get_extraction_value(extraction: Optional[Dict[str, Any]], key: str) -> str:
    if not extraction:
        return ""
    value = extraction.get(key, "")
    return value.strip() if isinstance(value, str) else ""


def build_cache_hit_answer(
    cached_answer: str,
    intent: Intent,
    extraction: Optional[Dict[str, Any]] = None,
) -> tuple[str, Dict[str, Any]]:
    """
    Phase 2.7 cache-hit paraphrase.

    Không gọi LLM Generator.
    Chỉ dùng template nhẹ để giữ latency thấp.
    """

    metadata: Dict[str, Any] = {
        "paraphrased": False,
        "paraphrase_strategy": "none",
    }

    if not cached_answer.strip():
        return cached_answer, metadata

    # FAQ nên giữ nguyên vì thông tin ổn định: wifi, giờ mở cửa, thanh toán...
    if intent == Intent.FAQ:
        metadata["paraphrase_strategy"] = "faq_keep_original"
        return cached_answer, metadata

    # Ignore cũng giữ nguyên để tránh phản hồi quá đà.
    if intent == Intent.IGNORE:
        metadata["paraphrase_strategy"] = "ignore_keep_original"
        return cached_answer, metadata

    # Order không cache theo policy, nhưng nếu lỡ gọi vào đây thì vẫn giữ nguyên.
    if intent == Intent.ORDER:
        metadata["paraphrase_strategy"] = "order_keep_original"
        return cached_answer, metadata

    if intent != Intent.CONSULTANT:
        return cached_answer, metadata

    context = _get_extraction_value(extraction, "context")
    cache_key = _get_extraction_value(extraction, "cache_key")

    prefix = ""

    if "giá mềm" in context or "rẻ" in context or "budget" in context:
        if "dễ uống" in context or "easy" in context:
            prefix = "Dạ, với tiêu chí giá mềm và dễ uống, "
        else:
            prefix = "Dạ, với tiêu chí giá mềm, "
    elif "dễ uống" in context:
        prefix = "Dạ, nếu anh/chị muốn món dễ uống, "
    elif "gợi ý" in cache_key:
        prefix = "Dạ, em gợi ý như sau: "

    if not prefix:
        metadata["paraphrase_strategy"] = "consultant_keep_original"
        return cached_answer, metadata

    lowered = cached_answer.lower()
    if lowered.startswith(("dạ", "với tiêu chí", "nếu anh", "nếu chị")):
        metadata["paraphrase_strategy"] = "consultant_already_natural"
        return cached_answer, metadata

    answer = prefix + cached_answer[0].lower() + cached_answer[1:]

    metadata.update(
        {
            "paraphrased": True,
            "paraphrase_strategy": "consultant_context_prefix",
            "context": context,
            "cache_key": cache_key,
        }
    )

    return answer, metadata