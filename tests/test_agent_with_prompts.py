import asyncio
from dataclasses import dataclass
from typing import Optional

from app.core.constants import Intent
from app.core.schemas import ChatRequest
from app.services.chat_service import chat_service


@dataclass
class TestCase:
    name: str
    text: str
    expected_intent: Intent
    must_contain: list[str]
    must_not_contain: list[str]
    max_length: Optional[int] = None


TEST_CASES = [
    # ======================
    # ORDER
    # ======================
    TestCase(
        name="order_bac_xiu",
        text="Cho anh một ly bạc xỉu đá",
        expected_intent=Intent.ORDER,
        must_contain=[
            "Bạc xỉu",
            "54.000",
            "xác nhận",
        ],
        must_not_contain=[
            "đơn hàng đã được xác nhận",
            "tổng tiền",
        ],
    ),

    # ======================
    # FAQ
    # ======================
    TestCase(
        name="faq_wifi",
        text="Wifi quán là gì?",
        expected_intent=Intent.FAQ,
        must_contain=[
            "Highlands_Guest",
            "highlands123",
        ],
        must_not_contain=[
            "tôi đoán",
            "có thể là",
        ],
    ),

    # ======================
    # CONSULTANT
    # ======================
    TestCase(
        name="consult_less_sweet",
        text="Có món nào ít ngọt không?",
        expected_intent=Intent.CONSULTANT,
        must_contain=[
            "ít ngọt",
        ],
        must_not_contain=[
            "5 món",
            "10 món",
            "không biết",
        ],
        max_length=500,
    ),

    # ======================
    # IGNORE
    # ======================
    TestCase(
        name="ignore_noise",
        text="haha",
        expected_intent=Intent.IGNORE,
        must_contain=[
            "đặt món",
        ],
        must_not_contain=[
            "wifi",
            "54.000",
        ],
    ),
]


async def run_case(case: TestCase):
    response = await chat_service.chat(
        ChatRequest(
            text=case.text,
            session_id=f"test-{case.name}",
        )
    )

    errors = []

    # intent
    if response.intent != case.expected_intent:
        errors.append(
            f"Intent mismatch: "
            f"{response.intent} != {case.expected_intent}"
        )

    answer_lower = response.answer.lower()

    # must contain
    for keyword in case.must_contain:
        if keyword.lower() not in answer_lower:
            errors.append(
                f"Missing keyword: {keyword}"
            )

    # must not contain
    for keyword in case.must_not_contain:
        if keyword.lower() in answer_lower:
            errors.append(
                f"Forbidden keyword: {keyword}"
            )

    # max length
    if case.max_length:
        if len(response.answer) > case.max_length:
            errors.append(
                f"Too long: {len(response.answer)} chars"
            )

    return {
        "name": case.name,
        "passed": len(errors) == 0,
        "errors": errors,
        "response": response.answer,
    }


async def main():
    print("=" * 80)
    print("AGENT TEST")
    print("=" * 80)

    passed = 0

    for case in TEST_CASES:
        result = await run_case(case)

        if result["passed"]:
            passed += 1
            status = "PASS"
        else:
            status = "FAIL"

        print("\n" + "-" * 80)
        print(f"[{status}] {result['name']}")
        print(f"Response: {result['response']}")

        if result["errors"]:
            print("Errors:")
            for err in result["errors"]:
                print(f"  - {err}")

    print("\n" + "=" * 80)
    print(f"RESULT: {passed}/{len(TEST_CASES)} passed")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())