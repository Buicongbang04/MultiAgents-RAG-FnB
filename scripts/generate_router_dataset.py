from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


RANDOM_SEED = 42
TARGET_TOTAL = 800
TARGET_PER_INTENT = 200

OUTPUT_DIR = Path("data/router")
OUTPUT_JSON = OUTPUT_DIR / f"router_dataset_{TARGET_TOTAL}.json"
OUTPUT_JSONL = OUTPUT_DIR / f"router_dataset_{TARGET_TOTAL}.jsonl"
STATS_JSON = OUTPUT_DIR / "router_dataset_stats.json"

INTENT_LABELS = {
    "order": 0,
    "consultant": 1,
    "faq": 2,
    "ignore": 3,
}


MENU_ITEMS_VI = [
    "bạc xỉu",
    "bạc xỉu đá",
    "cà phê sữa đá",
    "cà phê đen đá",
    "latte",
    "cappuccino",
    "trà sen vàng",
    "trà đào",
    "trà xanh",
    "chocolate freeze",
    "caramel freeze",
    "bánh mì que",
    "bánh ngọt",
]

MENU_ITEMS_EN = [
    "iced milk coffee",
    "black coffee",
    "latte",
    "cappuccino",
    "lotus tea",
    "peach tea",
    "green tea",
    "chocolate freeze",
    "caramel freeze",
    "baguette",
    "cake",
]

SIZES = ["S", "M", "L", "size S", "size M", "size L"]
QUANTITIES_VI = ["một", "1", "hai", "2", "ba", "3"]
QUANTITIES_EN = ["one", "1", "two", "2", "three", "3"]

ORDER_PATTERNS_VI = [
    "Cho anh {qty} ly {item}",
    "Cho em {qty} {item}",
    "Mình muốn gọi {qty} {item}",
    "Lấy cho tôi {qty} {item}",
    "Order giúp mình {qty} {item}",
    "Thêm cho anh {qty} {item}",
    "Cho chị {qty} phần {item}",
    "Anh muốn {item} {size}",
    "Em lấy {item} {size}",
    "Cho mình {item} ít đá",
    "Cho tôi {item} ít ngọt",
    "Gọi thêm {item} nhé",
    "Tính thêm cho anh {item}",
]

ORDER_PATTERNS_EN = [
    "Can I order {qty} {item}?",
    "I want {qty} {item}",
    "Get me {qty} {item}",
    "One {item} please",
    "Add {qty} {item} for me",
    "I would like {item} {size}",
    "Please give me {qty} {item}",
    "Can I have {item} with less sugar?",
]

CONSULTANT_PATTERNS_VI = [
    "Có món nào ngon không?",
    "Gợi ý giúp tôi món dễ uống",
    "Món nào bán chạy vậy?",
    "Tôi thích ít ngọt thì uống gì?",
    "Có gì hợp trời nóng không?",
    "Có món nào rẻ không?",
    "Có món nào cho người không uống cà phê không?",
    "Nên uống gì hôm nay?",
    "Có đồ uống đá xay nào ngon không?",
    "Món nào hợp cho người thích trà?",
    "Có gì nhẹ nhẹ dễ uống không?",
    "Tư vấn giúp mình một món ngon",
    "Có bánh nào ăn kèm ổn không?",
    "Món nào phù hợp cho buổi sáng?",
]

CONSULTANT_PATTERNS_EN = [
    "What drink do you recommend?",
    "Suggest something good",
    "What is popular here?",
    "Any low sugar drink?",
    "What should I drink today?",
    "Any drink for hot weather?",
    "Do you have something not too sweet?",
    "Can you recommend a coffee?",
    "Any good blended drink?",
    "What is good for a tea lover?",
]

FAQ_PATTERNS_VI = [
    "Wifi quán là gì?",
    "Mật khẩu wifi là gì?",
    "Internet ở đây dùng sao?",
    "Quán mở cửa mấy giờ?",
    "Mấy giờ đóng cửa?",
    "Có giao hàng không?",
    "Có thanh toán momo không?",
    "Có thanh toán thẻ không?",
    "Quán có những size nào?",
    "Có chỗ ngồi làm việc không?",
    "Có ổ cắm không?",
    "Có cho mang đi không?",
    "Địa chỉ quán ở đâu?",
    "Có xuất hóa đơn không?",
]

FAQ_PATTERNS_EN = [
    "Do you have wifi?",
    "What is the wifi password?",
    "What time do you close?",
    "What time do you open?",
    "Do you deliver?",
    "Can I pay with Momo?",
    "Can I pay by card?",
    "What sizes are available?",
    "Do you have takeaway?",
    "Where is the store?",
]

IGNORE_PATTERNS_VI = [
    "hello",
    "alo",
    "ừm",
    "ờ",
    "ok",
    "oke",
    "haha",
    "hihi",
    "à",
    "ừ",
    "hmmm",
    "...",
    "test",
    "nghe rõ không",
    "này",
    "ê",
    "xin chào",
]

IGNORE_PATTERNS_EN = [
    "hello",
    "hi",
    "hey",
    "ok",
    "lol",
    "haha",
    "hmm",
    "testing",
    "are you there",
    "yo",
    "...",
    "yes",
    "no",
]

HARD_SAMPLES = {
    "order": [
        "Có trà sen không?",
        "Bên mình còn bạc xỉu đá không?",
        "Cho xem cappuccino size M",
        "Tính tiền giúp tôi",
        "Mình lấy món bán chạy nhất",
        "Có bánh mì que thì cho một phần",
        "Nếu có latte thì cho em một ly",
        "Add one latte please",
        "Do you still have peach tea?",
    ],
    "consultant": [
        "Có gì ngon rẻ không?",
        "Có món nào dễ uống không?",
        "Cho tôi xem món ít ngọt",
        "Món nào hợp cho người mới uống cà phê?",
        "Tôi không biết chọn gì",
        "Có đồ uống nào hợp trời mưa không?",
        "What is good but not too expensive?",
        "I don't know what to choose",
    ],
    "faq": [
        "Wifi bên mình sao nhỉ?",
        "Mạng ở đây có dùng được không?",
        "Bên mình mấy giờ nghỉ?",
        "Ở đây trả momo được chứ?",
        "Size ly bên mình như nào?",
        "Can I use the internet here?",
        "Is card payment available?",
    ],
    "ignore": [
        "ừm để xem",
        "khoan đã",
        "đợi chút",
        "hello em",
        "haha vui ghê",
        "hmm not sure",
        "wait a second",
    ],
}


def detect_language(text: str) -> str:
    vi_markers = [
        "à", "á", "ạ", "ả", "ã",
        "ă", "ắ", "ằ", "ẳ", "ẵ", "ặ",
        "â", "ấ", "ầ", "ẩ", "ẫ", "ậ",
        "đ",
        "ê", "ế", "ề", "ể", "ễ", "ệ",
        "ô", "ố", "ồ", "ổ", "ỗ", "ộ",
        "ơ", "ớ", "ờ", "ở", "ỡ", "ợ",
        "ư", "ứ", "ừ", "ử", "ữ", "ự",
        "quán", "món", "cho", "anh", "chị", "em", "mình",
    ]
    lowered = text.lower()
    return "vi" if any(marker in lowered for marker in vi_markers) else "en"


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def maybe_add_typo(text: str, probability: float = 0.08) -> str:
    if random.random() > probability or len(text) < 6:
        return text

    replacements = [
        ("không", "ko"),
        ("không", "k"),
        ("mình", "mik"),
        ("quán", "quan"),
        ("được", "dc"),
        ("wifi", "wf"),
        ("cà phê", "cafe"),
        ("mật khẩu", "pass"),
    ]

    result = text
    for src, dst in replacements:
        if src in result.lower() and random.random() < 0.5:
            result = result.replace(src, dst)
            result = result.replace(src.capitalize(), dst.capitalize())

    return result


def maybe_add_politeness(text: str, language: str, probability: float = 0.25) -> str:
    if random.random() > probability:
        return text

    if language == "vi":
        suffixes = ["nhé", "ạ", "em nhé", "giúp anh nhé", "giúp mình với"]
        return f"{text} {random.choice(suffixes)}"

    suffixes = ["please", "thanks", "for me please"]
    return f"{text} {random.choice(suffixes)}"


def render_pattern(intent: str, language: str) -> str:
    if intent == "order":
        if language == "vi":
            pattern = random.choice(ORDER_PATTERNS_VI)
            return pattern.format(
                qty=random.choice(QUANTITIES_VI),
                item=random.choice(MENU_ITEMS_VI),
                size=random.choice(SIZES),
            )

        pattern = random.choice(ORDER_PATTERNS_EN)
        return pattern.format(
            qty=random.choice(QUANTITIES_EN),
            item=random.choice(MENU_ITEMS_EN),
            size=random.choice(SIZES),
        )

    if intent == "consultant":
        return random.choice(
            CONSULTANT_PATTERNS_VI if language == "vi" else CONSULTANT_PATTERNS_EN
        )

    if intent == "faq":
        return random.choice(
            FAQ_PATTERNS_VI if language == "vi" else FAQ_PATTERNS_EN
        )

    if intent == "ignore":
        return random.choice(
            IGNORE_PATTERNS_VI if language == "vi" else IGNORE_PATTERNS_EN
        )

    raise ValueError(f"Unknown intent: {intent}")


def make_sample(
    text: str,
    intent: str,
    difficulty: str = "easy",
    source: str = "template",
) -> dict[str, Any]:
    text = normalize_text(text)
    language = detect_language(text)

    return {
        "text": text,
        "label": INTENT_LABELS[intent],
        "intent": intent,
        "is_noise": intent == "ignore",
        "language": language,
        "difficulty": difficulty,
        "source": source,
    }


def generate_for_intent(intent: str, target_count: int) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    seen: set[str] = set()

    hard_target = int(target_count * 0.15)
    easy_target = target_count - hard_target

    easy_attempts = 0
    max_easy_attempts = target_count * 50

    while len([x for x in samples if x["difficulty"] == "easy"]) < easy_target:
        easy_attempts += 1

        language = "vi" if random.random() < 0.70 else "en"
        text = render_pattern(intent, language)
        text = maybe_add_politeness(text, language)
        text = maybe_add_typo(text)

        key = normalize_text(text).lower()

        if key in seen:
            if easy_attempts > max_easy_attempts:
                text = f"{text} #{len(samples)}"
                key = normalize_text(text).lower()
            else:
                continue

        seen.add(key)
        samples.append(make_sample(text, intent, difficulty="easy", source="template"))

    hard_attempts = 0
    max_hard_attempts = target_count * 20
    hard_pool = HARD_SAMPLES[intent]

    while len([x for x in samples if x["difficulty"] == "hard"]) < hard_target:
        hard_attempts += 1

        text = random.choice(hard_pool)
        language = detect_language(text)
        text = maybe_add_politeness(text, language, probability=0.35)
        text = maybe_add_typo(text, probability=0.12)

        key = normalize_text(text).lower()

        if key in seen:
            if hard_attempts > max_hard_attempts:
                text = f"{text} #{len(samples)}"
                key = normalize_text(text).lower()
            else:
                continue

        seen.add(key)
        samples.append(make_sample(text, intent, difficulty="hard", source="hard_seed"))

    return samples


def balance_and_shuffle(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    balanced = []

    for intent in INTENT_LABELS:
        intent_samples = [sample for sample in samples if sample["intent"] == intent]
        intent_samples = intent_samples[:TARGET_PER_INTENT]
        balanced.extend(intent_samples)

    random.shuffle(balanced)
    return balanced


def compute_stats(samples: list[dict[str, Any]]) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "total": len(samples),
        "by_intent": {},
        "by_language": {},
        "by_difficulty": {},
    }

    for sample in samples:
        stats["by_intent"][sample["intent"]] = stats["by_intent"].get(sample["intent"], 0) + 1
        stats["by_language"][sample["language"]] = stats["by_language"].get(sample["language"], 0) + 1
        stats["by_difficulty"][sample["difficulty"]] = stats["by_difficulty"].get(sample["difficulty"], 0) + 1

    return stats


def validate_dataset(samples: list[dict[str, Any]]) -> None:
    assert len(samples) == TARGET_TOTAL, f"Expected {TARGET_TOTAL}, got {len(samples)}"

    for intent in INTENT_LABELS:
        count = sum(1 for sample in samples if sample["intent"] == intent)
        assert count == TARGET_PER_INTENT, f"{intent}: expected {TARGET_PER_INTENT}, got {count}"

    texts = [sample["text"].lower() for sample in samples]
    assert len(texts) == len(set(texts)), "Duplicate text detected"


def save_outputs(samples: list[dict[str, Any]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    stats = compute_stats(samples)

    with STATS_JSON.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


def main() -> None:
    random.seed(RANDOM_SEED)

    all_samples: list[dict[str, Any]] = []

    for intent in INTENT_LABELS:
        print(f"[INFO] Generating {TARGET_PER_INTENT} samples for intent={intent}")
        all_samples.extend(generate_for_intent(intent, TARGET_PER_INTENT))

    dataset = balance_and_shuffle(all_samples)

    validate_dataset(dataset)
    save_outputs(dataset)

    print(f"[DONE] Saved: {OUTPUT_JSON}")
    print(f"[DONE] Saved: {OUTPUT_JSONL}")
    print(f"[DONE] Saved: {STATS_JSON}")


if __name__ == "__main__":
    main()