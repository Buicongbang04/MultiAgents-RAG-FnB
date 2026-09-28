from app.core.config import get_settings
from app.core.constants import SourceType


def format_vnd(price: int) -> str:
    return f"{price:,}".replace(",", ".") + "đ"


def short_description(description: str | None) -> str:
    """"Trà X size M (59.000đ) — chua nhẹ, thanh mát. Phù hợp…" → "chua nhẹ, thanh mát"."""
    if not description:
        return ""
    text = description.split("—", 1)[-1].strip()
    return text.split(".", 1)[0].strip()


def format_sizes(variants: list[dict]) -> str:
    """"S 49.000đ · M 59.000đ · L 69.000đ" — món một size chỉ hiện giá."""
    if len(variants) == 1:
        return format_vnd(variants[0]["price"])
    return " · ".join(f"{v['size']} {format_vnd(v['price'])}" for v in variants)


def format_price_range(variants: list[dict]) -> str:
    """"45.000–65.000đ" cho món nhiều size, "39.000đ" cho món một size."""
    if not variants:
        return ""
    prices = [v["price"] for v in variants]
    if min(prices) == max(prices):
        return format_vnd(prices[0])
    return f"{format_vnd(min(prices))[:-1]}–{format_vnd(max(prices))}"


def build_order_structured_context(resolution) -> str:
    """Context cho LLM: trạng thái đơn + NEXT_ACTION đã quyết định tất định."""
    from app.agents.order_resolution import OrderAction

    if resolution.action == OrderAction.NO_MATCH:
        return ""

    lines = ["ORDER_STATE:"]
    if resolution.action == OrderAction.ASK_ITEM:
        lines.append("- item: (chưa rõ)")
        lines.append("CANDIDATE_ITEMS:")
        lines.extend(
            f"- {name}: {format_price_range(resolution.candidate_variants.get(name) or [])}"
            for name in resolution.candidates
        )
    else:
        lines.append(f"- item: {resolution.item_name}")
        if resolution.category:
            lines.append(f"- category: {resolution.category}")
        lines.append("AVAILABLE_SIZES:")
        lines.extend(f"- {v['size']}: {format_vnd(v['price'])}" for v in resolution.variants)
        if resolution.variants:
            lines.append(f"- description: {short_description(resolution.variants[0]['description'])}")

    if resolution.availability_question:
        lines.append("- question_type: availability (khách hỏi quán có bán món này không → mở đầu bằng 'Dạ có ạ')")
    lines.append(f"- quantity: {resolution.quantity}")
    lines.append(f"- requested_size: {resolution.size or '(khách chưa chọn)'}")
    if resolution.selected:
        lines.append(
            f"- selected: {resolution.selected['size']} — {format_vnd(resolution.selected['price'])}/ly"
        )

    instructions = {
        OrderAction.ASK_ITEM: "ASK_ITEM — hỏi khách muốn món nào trong CANDIDATE_ITEMS.",
        OrderAction.ASK_SIZE: "ASK_SIZE — liệt kê các size kèm giá trong AVAILABLE_SIZES và hỏi khách chọn size nào.",
        OrderAction.SIZE_UNAVAILABLE: "SIZE_UNAVAILABLE — báo size khách chọn không có, mời chọn size trong AVAILABLE_SIZES.",
        OrderAction.CONFIRM: "CONFIRM — nhắc lại số lượng, tên món, size, giá mỗi ly và xin khách xác nhận.",
    }
    lines.extend(
        [
            "",
            f"NEXT_ACTION: {instructions[resolution.action]}",
            "",
            "ORDER_RULES:",
            "- Follow NEXT_ACTION exactly; ask only one question.",
            "- Never pick a size for the customer.",
            "- Do not confirm the order as completed.",
            "- Do not calculate total price.",
            "- Do not invent item, size, price, topping, or discount.",
        ]
    )
    return "\n".join(lines)


def build_consultant_structured_context(sources, variants: dict[str, list[dict]]) -> str:
    """Gộp các size của cùng một món thành một dòng (mỗi size là một node riêng)."""
    if not sources:
        return ""

    context_config = get_settings().rag.context
    lines = ["RECOMMENDABLE_MENU_ITEMS:"]

    for name, item_variants in list(variants.items())[: context_config.consultant_max_menu_items]:
        if not item_variants:
            continue
        line = (
            f"- item: {name} | sizes: {format_sizes(item_variants)}"
            f" | category: {item_variants[0]['category']}"
        )
        description = short_description(item_variants[0]["description"])
        if description:
            line += f" | note: {description}"
        lines.append(line)

    lines.append("")
    lines.append("SUPPORTING_CONTEXT:")

    supporting = [source for source in sources if source.source_type != SourceType.MENU]
    for source in supporting[: context_config.consultant_max_supporting]:
        lines.append(f"- {source.text}")

    lines.extend(
        [
            "",
            "CONSULTANT_RULES:",
            "- Recommend maximum 3 distinct items; mention each item only once with its sizes.",
            "- Only recommend items listed in RECOMMENDABLE_MENU_ITEMS.",
            "- If user preference is unclear, ask one short follow-up question.",
            "- Do not invent menu items, prices, promotions, or policies.",
        ]
    )

    return "\n".join(lines)


def build_faq_structured_context(sources) -> str:
    if not sources:
        return ""

    lines = ["FAQ_CONTEXT:"]

    max_items = get_settings().rag.context.faq_max_items
    for idx, source in enumerate(sources[:max_items], start=1):
        lines.append(f"{idx}. {source.text}")

    lines.extend(
        [
            "",
            "FAQ_RULES:",
            "- Answer only from FAQ_CONTEXT.",
            "- If the answer is not present, say the information was not found.",
            "- Do not infer missing store policy, price, promotion, or opening hours.",
            "- Keep the answer short and clear.",
        ]
    )

    return "\n".join(lines)