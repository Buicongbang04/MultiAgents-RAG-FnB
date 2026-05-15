from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.router_agent import RouterAgent

import asyncio
from app.core.schemas import RouterInput

TEST_META_PATH = Path("data/router/sft/test_with_meta.jsonl")

INTENTS = ["order", "consultant", "faq", "ignore"]


def load_jsonl(path: Path) -> list[dict]:
    assert path.exists(), f"File not found: {path}"

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def normalize_prediction(prediction) -> str:
    if isinstance(prediction, str):
        return prediction.strip().lower()

    if isinstance(prediction, dict):
        if "action" in prediction:
            value = prediction["action"]
            return getattr(value, "value", value).strip().lower()

        if "intent" in prediction:
            value = prediction["intent"]
            return getattr(value, "value", value).strip().lower()

    if hasattr(prediction, "action"):
        value = prediction.action
        return getattr(value, "value", value).strip().lower()

    if hasattr(prediction, "intent"):
        value = prediction.intent
        return getattr(value, "value", value).strip().lower()

    return str(prediction).strip().lower()


async def call_router(router: RouterAgent, text: str, idx: int) -> str:
    router_input = RouterInput(
        session_id=f"eval-{idx}",
        text=text,
    )

    if hasattr(router, "classify"):
        output = await router.classify(router_input)
        return normalize_prediction(output)

    raise AttributeError("RouterAgent must expose classify(router_input)")

def safe_intent(intent: str) -> str:
    if intent in INTENTS:
        return intent
    return "parse_error"


def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    metrics = {}

    total = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)

    metrics["accuracy"] = correct / total if total else 0.0

    per_class = {}

    for intent in INTENTS:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == intent and p == intent)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != intent and p == intent)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == intent and p != intent)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        per_class[intent] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(1 for t in y_true if t == intent),
        }

    metrics["per_class"] = per_class
    return metrics


def build_confusion_matrix(y_true: list[str], y_pred: list[str]) -> dict:
    labels = INTENTS + ["parse_error"]
    matrix = {true: {pred: 0 for pred in labels} for true in INTENTS}

    for true, pred in zip(y_true, y_pred):
        pred = safe_intent(pred)
        matrix[true][pred] += 1

    return matrix


def print_confusion_matrix(matrix: dict) -> None:
    headers = INTENTS + ["parse_error"]

    print("\nCONFUSION MATRIX")
    print("-" * 80)
    print(f"{'true/pred':<14}" + "".join(f"{h:<14}" for h in headers))

    for true in INTENTS:
        row = matrix[true]
        print(f"{true:<14}" + "".join(f"{row[h]:<14}" for h in headers))


async def main() -> None:
    rows = load_jsonl(TEST_META_PATH)
    router = RouterAgent()

    y_true = []
    y_pred = []
    errors = []

    difficulty_total = Counter()
    difficulty_correct = Counter()

    language_total = Counter()
    language_correct = Counter()

    for sample in rows:
        text = sample["text"]
        true_intent = sample["intent"]

        pred_intent = await call_router(router, text, len(y_true))
        pred_intent = safe_intent(pred_intent)

        y_true.append(true_intent)
        y_pred.append(pred_intent)

        difficulty = sample.get("difficulty", "unknown")
        language = sample.get("language", "unknown")

        difficulty_total[difficulty] += 1
        language_total[language] += 1

        if pred_intent == true_intent:
            difficulty_correct[difficulty] += 1
            language_correct[language] += 1
        else:
            errors.append(
                {
                    "text": text,
                    "true": true_intent,
                    "pred": pred_intent,
                    "difficulty": difficulty,
                    "language": language,
                    "source": sample.get("source"),
                }
            )

    metrics = compute_metrics(y_true, y_pred)
    matrix = build_confusion_matrix(y_true, y_pred)

    print("=" * 80)
    print("ROUTER BASELINE EVALUATION")
    print("=" * 80)
    print(f"Test samples: {len(rows)}")
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")

    print("\nPER-CLASS METRICS")
    print("-" * 80)
    print(f"{'intent':<14}{'precision':<14}{'recall':<14}{'f1':<14}{'support':<14}")

    for intent, item in metrics["per_class"].items():
        print(
            f"{intent:<14}"
            f"{item['precision']:<14.4f}"
            f"{item['recall']:<14.4f}"
            f"{item['f1']:<14.4f}"
            f"{item['support']:<14}"
        )

    print_confusion_matrix(matrix)

    print("\nDIFFICULTY ACCURACY")
    print("-" * 80)
    for difficulty, total in difficulty_total.items():
        correct = difficulty_correct[difficulty]
        acc = correct / total if total else 0.0
        print(f"{difficulty}: {correct}/{total} = {acc:.2%}")

    print("\nLANGUAGE ACCURACY")
    print("-" * 80)
    for language, total in language_total.items():
        correct = language_correct[language]
        acc = correct / total if total else 0.0
        print(f"{language}: {correct}/{total} = {acc:.2%}")

    print("\nERROR SAMPLES")
    print("-" * 80)
    for idx, err in enumerate(errors[:30], start=1):
        print(
            f"{idx}. [{err['language']} | {err['difficulty']}] "
            f"true={err['true']} pred={err['pred']} | {err['text']}"
        )

    print(f"\nTotal errors: {len(errors)}")


if __name__ == "__main__":
    asyncio.run(main())