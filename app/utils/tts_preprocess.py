import re


def preprocess_tts_text(text: str) -> str:
    if not text:
        return text

    output = text

    # 49.000đ / 49,000đ -> 49k
    output = re.sub(
        r"(\d{1,3})[.,](\d{3})\s*(đ|vnd|VND|VNĐ)",
        lambda m: f"{m.group(1)}k",
        output,
    )

    # 54000 VND -> 54k
    output = re.sub(
        r"\b(\d{2,3})000\s*(đ|vnd|VND|VNĐ)\b",
        lambda m: f"{m.group(1)}k",
        output,
    )

    # VAT -> thuế
    output = re.sub(r"\bVAT\b", "thuế", output, flags=re.IGNORECASE)

    # size M/L/S đọc thân thiện hơn
    output = re.sub(r"\bsize\s+S\b", "size ét", output, flags=re.IGNORECASE)
    output = re.sub(r"\bsize\s+M\b", "size mờ", output, flags=re.IGNORECASE)
    output = re.sub(r"\bsize\s+L\b", "size lờ", output, flags=re.IGNORECASE)

    return output