from app.core.constants import SourceType


def format_vnd(price: int) -> str:
    return f"{price:,}".replace(",", ".") + "đ"


def get_item_name(text: str) -> str:
    return text.split("|")[0].strip()


def build_order_structured_context(user_text: str, sources) -> str:
    if not sources:
        return ""

    user_text_lower = user_text.lower()

    exact_sources = [
        source for source in sources
        if get_item_name(source.text).lower() in user_text_lower
    ]

    best_source = exact_sources[0] if exact_sources else sources[0]

    lines = [
        "BEST_MATCH:",
        f"- id: {best_source.source_id}",
        f"- item: {get_item_name(best_source.text)}",
    ]

    price = best_source.metadata.get("price")
    size = best_source.metadata.get("size")
    category = best_source.metadata.get("category")

    if price:
        lines.append(f"- price: {format_vnd(price)}")
    if size:
        lines.append(f"- size: {size}")
    if category:
        lines.append(f"- category: {category}")

    lines.append("")
    lines.append("ALTERNATIVES:")

    for source in sources:
        if source.source_id == best_source.source_id:
            continue

        alt_price = source.metadata.get("price")
        alt_size = source.metadata.get("size")

        line = f"- {get_item_name(source.text)}"
        if alt_size:
            line += f" | size: {alt_size}"
        if alt_price:
            line += f" | price: {format_vnd(alt_price)}"

        lines.append(line)

    lines.extend(
        [
            "",
            "ORDER_RULES:",
            "- Use BEST_MATCH as the primary item.",
            "- Do not confirm the order as completed.",
            "- Ask the customer to confirm before adding to cart.",
            "- Do not calculate total price.",
            "- Do not invent item, size, price, topping, or discount.",
        ]
    )

    return "\n".join(lines)


def build_consultant_structured_context(sources) -> str:
    if not sources:
        return ""

    menu_sources = [
        source for source in sources
        if source.source_type == SourceType.MENU
    ]

    faq_sources = [
        source for source in sources
        if source.source_type != SourceType.MENU
    ]

    lines = ["RECOMMENDABLE_MENU_ITEMS:"]

    for source in menu_sources[:5]:
        price = source.metadata.get("price")
        size = source.metadata.get("size")
        category = source.metadata.get("category")

        line = f"- item: {get_item_name(source.text)}"

        if price:
            line += f" | price: {format_vnd(price)}"
        if size:
            line += f" | size: {size}"
        if category:
            line += f" | category: {category}"

        lines.append(line)

    lines.append("")
    lines.append("SUPPORTING_CONTEXT:")

    for source in faq_sources[:3]:
        lines.append(f"- {source.text}")

    lines.extend(
        [
            "",
            "CONSULTANT_RULES:",
            "- Recommend maximum 3 items.",
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

    for idx, source in enumerate(sources[:5], start=1):
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