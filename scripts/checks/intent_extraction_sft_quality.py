from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

SFT_DIR = Path("data/intent_extraction/sft")

TRAIN_PATH = SFT_DIR / "train.jsonl"
VAL_PATH = SFT_DIR / "val.jsonl"
TEST_PATH = SFT_DIR / "test.jsonl"
TEST_META_PATH = SFT_DIR / "test_with_meta.jsonl"

VALID_INTENTS = {"order", "consultant", "faq", "ignore"}
VALID_LANGUAGES = {"vi", "en"}
TARGET_FIELDS = {
    "subject",
    "action",
    "context",
    "cache_key",
    "intent",
    "language",
}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    assert path.exists(), f"Missing file: {path}"
    rows: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            assert line, f"Empty line {line_no} in {path}"
            rows.append(json.loads(line))

    return rows


def test_sft_files_valid() -> None:
    train = load_jsonl(TRAIN_PATH)
    val = load_jsonl(VAL_PATH)
    test = load_jsonl(TEST_PATH)
    meta = load_jsonl(TEST_META_PATH)

    assert len(train) == 800
    assert len(val) == 100
    assert len(test) == 100
    assert len(meta) == 100

    seen_user_texts = set()

    for split_name, rows in {
        "train": train,
        "val": val,
        "test": test,
    }.items():
        by_intent: Dict[str, int] = {}

        for row in rows:
            assert "messages" in row
            messages = row["messages"]
            assert len(messages) == 3

            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"
            assert messages[2]["role"] == "assistant"

            user_text = messages[1]["content"]
            assert user_text not in seen_user_texts
            seen_user_texts.add(user_text)

            target = json.loads(messages[2]["content"])
            assert set(target.keys()) == TARGET_FIELDS
            assert target["intent"] in VALID_INTENTS
            assert target["language"] in VALID_LANGUAGES

            if target["intent"] in {"order", "consultant", "faq"}:
                assert target["action"].strip()
                assert target["cache_key"].strip()

            by_intent[target["intent"]] = by_intent.get(target["intent"], 0) + 1

        assert set(by_intent.keys()) == VALID_INTENTS, f"{split_name}: {by_intent}"

    print("INTENT EXTRACTION SFT QUALITY PASS")