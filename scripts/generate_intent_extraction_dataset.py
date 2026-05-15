from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

RANDOM_SEED = 42

TARGET_TOTAL = 1000
INTENTS = ["order", "consultant", "faq", "ignore"]
TARGET_PER_INTENT = TARGET_TOTAL // len(INTENTS)

OUTPUT_DIR = Path("data/intent_extraction")
OUTPUT_JSON = OUTPUT_DIR / "intent_extraction_dataset.json"
OUTPUT_JSONL = OUTPUT_DIR / "intent_extraction_dataset.jsonl"
STATS_JSON = OUTPUT_DIR / "intent_extraction_stats.json"

MENU_ITEMS_VI = [
    "bạc xỉu đá",
    "bạc xỉu",
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
]

SUBJECTS_VI = ["anh", "chị", "em", "mình", "tôi", "tui", "bọn em", "tụi em"]
SUBJECTS_EN = ["I", "we", "my friend and I"]

QUANTITIES_VI = ["một", "1", "hai", "2", "ba", "3"]
QUANTITIES_EN = ["one", "1", "two", "2", "three", "3"]


def norm(text: str) -> str:
    return " ".join(text.strip().split())


def strip_dup_suffix(text: str) -> str:
    return re.sub(r"\s+#\d+$", "", text).strip()


def extract_subject_vi(text: str) -> str:
    clean = strip_dup_suffix(text.lower())

    patterns = [
        r"\bcho\s+(anh|chị|em|mình|tôi|tui|bọn em|tụi em)\b",
        r"\b(anh|chị|em|mình|tôi|tui|bọn em|tụi em)\s+(muốn|cần|hỏi|xin|đặt|gọi|lấy|order|thích)\b",
        r"\bcho\s+(mình|tôi|tui)\b",
        r"\bcho\s+(anh|chị|em)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, clean)
        if match:
            return match.group(1)

    # "Gợi ý món ... cho mình" is explicit subject/requester.
    match = re.search(r"\bcho\s+(mình|anh|chị|em|tôi|tui|bọn em|tụi em)\b", clean)
    if match:
        return match.group(1)

    return ""


def extract_subject_en(text: str) -> str:
    clean = strip_dup_suffix(text)

    if re.search(r"\bmy friend and i\b", clean, flags=re.IGNORECASE):
        return "my friend and I"
    if re.search(r"\bwe\b", clean, flags=re.IGNORECASE):
        return "we"
    if re.search(r"\bi\b", clean, flags=re.IGNORECASE):
        return "I"

    return ""


def make_sample(
    *,
    text: str,
    intent: str,
    language: str,
    subject: str,
    action: str,
    context: str,
    cache_key: str,
    is_hard: bool,
    source_template: str,
) -> Dict[str, Any]:
    return {
        "text": norm(text),
        "intent": intent,
        "language": language,
        "subject": subject,
        "action": norm(action),
        "context": norm(context),
        "cache_key": norm(cache_key),
        "is_hard": is_hard,
        "source_template": source_template,
    }


def generate_order_vi() -> Dict[str, Any]:
    subject = random.choice(SUBJECTS_VI)
    qty = random.choice(QUANTITIES_VI)
    item = random.choice(MENU_ITEMS_VI)
    context = random.choice(["", "ít ngọt", "nhiều đá", "ít đá", "mang đi", "size M", "size L"])

    text = f"Cho {subject} {qty} ly {item}"
    if context:
        text += f" {context}"

    action = f"{qty} {item}"
    if context:
        action = f"{action} {context}"

    return make_sample(
        text=text,
        intent="order",
        language="vi",
        subject=subject,
        action=action,
        context=context,
        cache_key=action,
        is_hard=False,
        source_template="order_vi_basic",
    )


def generate_order_en() -> Dict[str, Any]:
    subject = random.choice(SUBJECTS_EN)
    qty = random.choice(QUANTITIES_EN)
    item = random.choice(MENU_ITEMS_EN)
    context = random.choice(["", "less sugar", "more ice", "less ice", "take away", "size M", "size L"])

    if subject == "I":
        text = f"Can I order {qty} {item}"
    else:
        text = f"Can {subject.lower()} order {qty} {item}"

    if context:
        text += f" {context}"

    action = f"{qty} {item}"
    if context:
        action = f"{action} {context}"

    return make_sample(
        text=text,
        intent="order",
        language="en",
        subject=subject,
        action=action,
        context=context,
        cache_key=action,
        is_hard=False,
        source_template="order_en_basic",
    )


FAQ_TEMPLATES = [
    {
        "texts_vi": [
            ("Wifi quán là gì?", ""),
            ("Cho em xin mật khẩu wifi với nha", "em"),
            ("Dạ em hỏi xíu pass wifi ở đây là gì vậy ạ?", "em"),
            ("Internet ở quán dùng sao em?", ""),
        ],
        "texts_en": [
            ("What is the wifi password?", ""),
            ("Can I get the wifi password?", "I"),
            ("Do you have wifi here?", ""),
            ("How can I connect to the internet here?", "I"),
        ],
        "action_vi": "hỏi mật khẩu wifi",
        "cache_key_vi": "mật khẩu wifi",
        "action_en": "ask wifi password",
        "cache_key_en": "wifi password",
        "topic": "faq_wifi",
    },
    {
        "texts_vi": [
            ("Quán mở cửa mấy giờ?", ""),
            ("Mấy giờ đóng cửa vậy em?", ""),
            ("Hôm nay quán còn mở không?", ""),
            ("Bên mình mấy giờ nghỉ?", ""),
        ],
        "texts_en": [
            ("What time do you open?", ""),
            ("What time do you close?", ""),
            ("Are you still open today?", ""),
            ("What are your opening hours?", ""),
        ],
        "action_vi": "hỏi giờ mở cửa đóng cửa",
        "cache_key_vi": "giờ mở cửa đóng cửa",
        "action_en": "ask opening hours",
        "cache_key_en": "opening hours",
        "topic": "faq_opening_hours",
    },
    {
        "texts_vi": [
            ("Có thanh toán momo không?", ""),
            ("Quán có nhận thẻ không?", ""),
            ("Em chuyển khoản được không?", "em"),
            ("Ở đây thanh toán bằng gì được?", ""),
        ],
        "texts_en": [
            ("Can I pay by card?", "I"),
            ("Do you accept Momo?", ""),
            ("Can I pay by bank transfer?", "I"),
            ("What payment methods do you accept?", ""),
        ],
        "action_vi": "hỏi phương thức thanh toán",
        "cache_key_vi": "phương thức thanh toán",
        "action_en": "ask payment methods",
        "cache_key_en": "payment methods",
        "topic": "faq_payment",
    },
    {
        "texts_vi": [
            ("Có giao hàng không?", ""),
            ("Bên mình có ship không?", ""),
            ("Có cho mang đi không?", ""),
            ("Đặt take away được không?", ""),
        ],
        "texts_en": [
            ("Do you deliver?", ""),
            ("Can I order takeaway?", "I"),
            ("Do you have take away?", ""),
            ("Can you ship my order?", ""),
        ],
        "action_vi": "hỏi chính sách giao hàng mang đi",
        "cache_key_vi": "giao hàng mang đi",
        "action_en": "ask delivery takeaway policy",
        "cache_key_en": "delivery takeaway policy",
        "topic": "faq_delivery",
    },
]


def generate_faq() -> Dict[str, Any]:
    template = random.choice(FAQ_TEMPLATES)
    language = "vi" if random.random() < 0.7 else "en"

    if language == "vi":
        text, subject = random.choice(template["texts_vi"])
        return make_sample(
            text=text,
            intent="faq",
            language="vi",
            subject=subject,
            action=template["action_vi"],
            context="",
            cache_key=template["cache_key_vi"],
            is_hard=False,
            source_template=template["topic"],
        )

    text, subject = random.choice(template["texts_en"])
    return make_sample(
        text=text,
        intent="faq",
        language="en",
        subject=subject,
        action=template["action_en"],
        context="",
        cache_key=template["cache_key_en"],
        is_hard=False,
        source_template=template["topic"],
    )


CONSULTANT_TEMPLATES = [
    {
        "texts_vi": [
            ("Có gì ngon rẻ không?", ""),
            ("Em gợi ý món nào giá mềm mà dễ uống?", "em"),
            ("Có món nào ngon mà không quá mắc không?", ""),
            ("Tư vấn giúp mình món dễ uống giá ổn với", "mình"),
        ],
        "texts_en": [
            ("Can you recommend something affordable and easy to drink?", ""),
            ("What is good but not too expensive?", ""),
            ("Suggest something budget friendly and easy to drink", ""),
            ("I want something affordable and easy to drink", "I"),
        ],
        "action_vi": "gợi ý món dễ uống giá mềm",
        "cache_key_vi": "gợi ý món ngon rẻ dễ uống",
        "action_en": "recommend affordable easy to drink item",
        "cache_key_en": "recommend affordable easy to drink item",
        "context_vi": "giá mềm, dễ uống",
        "context_en": "affordable, easy to drink",
        "topic": "consultant_budget_easy",
    },
    {
        "texts_vi": [
            ("Tôi thích ít ngọt thì uống gì?", "tôi"),
            ("Có món nào ít ngọt dễ uống không?", ""),
            ("Gợi ý món ít ngọt cho mình", "mình"),
            ("Em muốn món nào ít ngọt", "em"),
        ],
        "texts_en": [
            ("Any low sugar drink?", ""),
            ("Can you recommend something not too sweet?", ""),
            ("What should I drink if I want less sugar?", "I"),
            ("I want a less sweet drink", "I"),
        ],
        "action_vi": "gợi ý món ít ngọt",
        "cache_key_vi": "gợi ý món ít ngọt",
        "action_en": "recommend low sugar drink",
        "cache_key_en": "recommend low sugar drink",
        "context_vi": "ít ngọt",
        "context_en": "less sugar",
        "topic": "consultant_low_sugar",
    },
    {
        "texts_vi": [
            ("Có gì hợp trời nóng không?", ""),
            ("Trời nóng nên uống món nào?", ""),
            ("Gợi ý món mát mát cho hôm nay đi", ""),
            ("Em muốn món mát dễ uống cho trời nóng", "em"),
        ],
        "texts_en": [
            ("Any drink for hot weather?", ""),
            ("What should I drink on a hot day?", "I"),
            ("Recommend something refreshing for hot weather", ""),
            ("I want something refreshing for hot weather", "I"),
        ],
        "action_vi": "gợi ý món mát",
        "cache_key_vi": "gợi ý món mát phù hợp trời nóng",
        "action_en": "recommend refreshing drink",
        "cache_key_en": "recommend refreshing drink",
        "context_vi": "trời nóng",
        "context_en": "hot weather",
        "topic": "consultant_hot_weather",
    },
]


def generate_consultant() -> Dict[str, Any]:
    template = random.choice(CONSULTANT_TEMPLATES)
    language = "vi" if random.random() < 0.7 else "en"

    if language == "vi":
        text, subject = random.choice(template["texts_vi"])
        extra_context = random.choice(["", " cho buổi sáng", " tối nay", " cho nhóm 5 người"])
        text = f"{text}{extra_context}"

        context = template["context_vi"]
        if extra_context.strip():
            context = f"{context}, {extra_context.strip()}"

        return make_sample(
            text=text,
            intent="consultant",
            language="vi",
            subject=subject,
            action=template["action_vi"],
            context=context,
            cache_key=template["cache_key_vi"],
            is_hard=False,
            source_template=template["topic"],
        )

    text, subject = random.choice(template["texts_en"])
    extra_context = random.choice(["", " for this morning", " tonight", " for five people"])
    text = f"{text}{extra_context}"

    context = template["context_en"]
    if extra_context.strip():
        context = f"{context}, {extra_context.strip()}"

    return make_sample(
        text=text,
        intent="consultant",
        language="en",
        subject=subject,
        action=template["action_en"],
        context=context,
        cache_key=template["cache_key_en"],
        is_hard=False,
        source_template=template["topic"],
    )


IGNORE_TEXTS_VI = [
    "hello em",
    "ừm để anh xem",
    "khoan đã",
    "đợi chút nha",
    "haha vui ghê",
    "alo nghe rõ không",
    "ok em",
]

IGNORE_TEXTS_EN = [
    "hello",
    "wait a second",
    "hmm let me see",
    "haha",
    "ok",
    "are you there",
    "testing",
]


def generate_ignore() -> Dict[str, Any]:
    language = "vi" if random.random() < 0.7 else "en"
    text = random.choice(IGNORE_TEXTS_VI if language == "vi" else IGNORE_TEXTS_EN)
    return make_sample(
        text=text,
        intent="ignore",
        language=language,
        subject="",
        action=norm(text.lower()),
        context="",
        cache_key=norm(text.lower()),
        is_hard=False,
        source_template="ignore_noise",
    )


HARD_SAMPLES = [
    make_sample(
        text="Mai sinh nhật bạn, tầm 7h tối, nhóm em 5 người, có món nào dễ uống không?",
        intent="consultant",
        language="vi",
        subject="em",
        action="gợi ý món dễ uống",
        context="mai, sinh nhật bạn, 7h tối, nhóm 5 người",
        cache_key="gợi ý món dễ uống",
        is_hard=True,
        source_template="hard_consultant_context_heavy",
    ),
    make_sample(
        text="Có bạc xỉu nào ngon không cho anh một ly?",
        intent="order",
        language="vi",
        subject="anh",
        action="một ly bạc xỉu",
        context="hỏi món ngon",
        cache_key="một ly bạc xỉu",
        is_hard=True,
        source_template="hard_order_consultant_mix",
    ),
    make_sample(
        text="Dạ em hỏi xíu pass wifi ở đây là gì vậy ạ?",
        intent="faq",
        language="vi",
        subject="em",
        action="hỏi mật khẩu wifi",
        context="",
        cache_key="mật khẩu wifi",
        is_hard=True,
        source_template="hard_faq_polite_filler",
    ),
    make_sample(
        text="Có món nào budget friendly không em?",
        intent="consultant",
        language="vi",
        subject="",
        action="gợi ý món giá mềm",
        context="budget friendly",
        cache_key="gợi ý món ngon rẻ",
        is_hard=True,
        source_template="hard_codeswitch_budget",
    ),
    make_sample(
        text="Can I get something cheap and maybe one latte?",
        intent="order",
        language="en",
        subject="I",
        action="one latte",
        context="cheap recommendation",
        cache_key="one latte",
        is_hard=True,
        source_template="hard_en_order_consultant_mix",
    ),
    make_sample(
        text="Budget friendly mà dễ uống thì món nào ổn?",
        intent="consultant",
        language="vi",
        subject="",
        action="gợi ý món dễ uống giá mềm",
        context="budget friendly, dễ uống",
        cache_key="gợi ý món ngon rẻ dễ uống",
        is_hard=True,
        source_template="hard_codeswitch_budget_easy",
    ),
    make_sample(
        text="Có gì ngon không, cho anh một latte luôn",
        intent="order",
        language="vi",
        subject="anh",
        action="một latte",
        context="hỏi món ngon",
        cache_key="một latte",
        is_hard=True,
        source_template="hard_order_consultant_latte",
    ),
    make_sample(
        text="Tomorrow morning, can you recommend something refreshing for five people?",
        intent="consultant",
        language="en",
        subject="",
        action="recommend refreshing drink",
        context="hot weather, tomorrow morning, for five people",
        cache_key="recommend refreshing drink",
        is_hard=True,
        source_template="hard_en_consultant_long_context",
    ),
]


def generator_for_intent(intent: str):
    if intent == "order":
        return generate_order_vi if random.random() < 0.7 else generate_order_en
    if intent == "consultant":
        return generate_consultant
    if intent == "faq":
        return generate_faq
    if intent == "ignore":
        return generate_ignore
    raise ValueError(intent)


def validate_sample(sample: Dict[str, Any]) -> None:
    required = [
        "text",
        "intent",
        "language",
        "subject",
        "action",
        "context",
        "cache_key",
        "is_hard",
        "source_template",
    ]
    for key in required:
        assert key in sample, f"Missing key: {key}"

    assert sample["intent"] in INTENTS
    assert sample["language"] in {"vi", "en"}

    if sample["intent"] in {"order", "consultant", "faq"}:
        assert sample["action"].strip(), f"Empty action: {sample}"
        assert sample["cache_key"].strip(), f"Empty cache_key: {sample}"


def generate_dataset() -> List[Dict[str, Any]]:
    random.seed(RANDOM_SEED)

    samples: List[Dict[str, Any]] = []
    seen: set[str] = set()

    hard_by_intent: Dict[str, List[Dict[str, Any]]] = {intent: [] for intent in INTENTS}
    for hard in HARD_SAMPLES:
        hard_by_intent[hard["intent"]].append(hard)

    for intent in INTENTS:
        target = TARGET_PER_INTENT
        hard_target = max(1, int(target * 0.12))
        easy_target = target - hard_target

        for hard in hard_by_intent[intent][:hard_target]:
            key = hard["text"].lower()
            if key not in seen:
                validate_sample(hard)
                seen.add(key)
                samples.append(hard)

        while sum(1 for s in samples if s["intent"] == intent and not s["is_hard"]) < easy_target:
            sample = generator_for_intent(intent)()
            key = sample["text"].lower()

            if key in seen:
                sample["text"] = f'{sample["text"]} #{len(samples)}'
                key = sample["text"].lower()

            validate_sample(sample)
            seen.add(key)
            samples.append(sample)

        while sum(1 for s in samples if s["intent"] == intent) < target:
            sample = generator_for_intent(intent)()
            sample["is_hard"] = True
            sample["source_template"] = "auto_hard_fill"
            key = sample["text"].lower()

            if key in seen:
                sample["text"] = f'{sample["text"]} #{len(samples)}'
                key = sample["text"].lower()

            validate_sample(sample)
            seen.add(key)
            samples.append(sample)

    random.shuffle(samples)
    return samples


def compute_stats(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "total": len(samples),
        "by_intent": {},
        "by_language": {},
        "hard_count": 0,
        "hard_ratio": 0.0,
        "empty_action": 0,
        "empty_cache_key": 0,
    }

    for sample in samples:
        stats["by_intent"][sample["intent"]] = stats["by_intent"].get(sample["intent"], 0) + 1
        stats["by_language"][sample["language"]] = stats["by_language"].get(sample["language"], 0) + 1
        stats["hard_count"] += int(bool(sample["is_hard"]))
        stats["empty_action"] += int(not bool(sample["action"].strip()))
        stats["empty_cache_key"] += int(not bool(sample["cache_key"].strip()))

    stats["hard_ratio"] = round(stats["hard_count"] / max(1, len(samples)), 4)
    return stats


def save_outputs(samples: List[Dict[str, Any]]) -> None:
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
    samples = generate_dataset()
    assert len(samples) == TARGET_TOTAL, f"Expected {TARGET_TOTAL}, got {len(samples)}"

    for intent in INTENTS:
        count = sum(1 for s in samples if s["intent"] == intent)
        assert count == TARGET_PER_INTENT, f"{intent}: expected {TARGET_PER_INTENT}, got {count}"

    save_outputs(samples)
    print(f"[DONE] Saved {OUTPUT_JSON}")
    print(f"[DONE] Saved {OUTPUT_JSONL}")
    print(f"[DONE] Saved {STATS_JSON}")


if __name__ == "__main__":
    main()