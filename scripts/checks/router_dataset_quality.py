import json
from pathlib import Path
from collections import Counter


DATASET_PATH = Path("data/router/router_dataset_800.json")

EXPECTED_TOTAL = 800
EXPECTED_PER_INTENT = 200

REQUIRED_FIELDS = {
    "text",
    "label",
    "intent",
    "is_noise",
    "language",
    "difficulty",
    "source",
}

INTENT_LABELS = {
    "order": 0,
    "consultant": 1,
    "faq": 2,
    "ignore": 3,
}


def load_dataset():
    assert DATASET_PATH.exists(), f"Dataset not found: {DATASET_PATH}"

    with DATASET_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    data = load_dataset()

    assert isinstance(data, list), "Dataset must be a list"
    assert len(data) == EXPECTED_TOTAL, (
        f"Expected {EXPECTED_TOTAL} samples, got {len(data)}"
    )

    texts = []
    intents = Counter()
    labels = Counter()
    languages = Counter()
    difficulties = Counter()
    sources = Counter()

    for idx, sample in enumerate(data):
        missing = REQUIRED_FIELDS - set(sample.keys())
        assert not missing, f"Sample {idx} missing fields: {missing}"

        assert isinstance(sample["text"], str)
        assert sample["text"].strip(), f"Sample {idx} has empty text"

        assert sample["intent"] in INTENT_LABELS, f"Invalid intent at {idx}"
        assert sample["label"] == INTENT_LABELS[sample["intent"]], (
            f"Wrong label at {idx}: "
            f"{sample['intent']} should be {INTENT_LABELS[sample['intent']]}"
        )

        assert sample["language"] in {"vi", "en"}, f"Invalid language at {idx}"
        assert sample["difficulty"] in {"easy", "hard"}, f"Invalid difficulty at {idx}"

        if sample["intent"] == "ignore":
            assert sample["is_noise"] is True, f"Ignore sample {idx} must be noise"
        else:
            assert sample["is_noise"] is False, f"Non-ignore sample {idx} must not be noise"

        texts.append(sample["text"].lower().strip())
        intents[sample["intent"]] += 1
        labels[sample["label"]] += 1
        languages[sample["language"]] += 1
        difficulties[sample["difficulty"]] += 1
        sources[sample["source"]] += 1

    duplicate_count = len(texts) - len(set(texts))
    assert duplicate_count == 0, f"Found {duplicate_count} duplicate texts"

    for intent in INTENT_LABELS:
        assert intents[intent] == EXPECTED_PER_INTENT, (
            f"{intent} count != {EXPECTED_PER_INTENT}: {intents[intent]}"
        )

    hard_ratio = difficulties["hard"] / len(data)
    assert 0.05 <= hard_ratio <= 0.20, f"Hard ratio out of range: {hard_ratio:.2%}"

    vi_ratio = languages["vi"] / len(data)
    en_ratio = languages["en"] / len(data)

    assert vi_ratio >= 0.55, f"Vietnamese ratio too low: {vi_ratio:.2%}"
    assert en_ratio >= 0.20, f"English ratio too low: {en_ratio:.2%}"

    print("=" * 80)
    print("ROUTER DATASET QUALITY CHECK PASSED")
    print("=" * 80)
    print(f"Total: {len(data)}")
    print(f"Intents: {dict(intents)}")
    print(f"Labels: {dict(labels)}")
    print(f"Languages: {dict(languages)}")
    print(f"Difficulties: {dict(difficulties)}")
    print(f"Sources: {dict(sources)}")
    print(f"Hard ratio: {hard_ratio:.2%}")
    print(f"VI ratio: {vi_ratio:.2%}")
    print(f"EN ratio: {en_ratio:.2%}")


if __name__ == "__main__":
    main()