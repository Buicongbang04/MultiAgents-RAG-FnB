"""Unit tests for the rule-based router — guards the word-boundary keyword fix."""
from app.agents.intent_rules import _keyword_matches, classify_by_rules
from app.core.constants import Intent


def test_keyword_word_boundary():
    # "add" must not match inside "address".
    assert _keyword_matches("what is your address", "address")
    assert not _keyword_matches("what is your address", "add")
    # phrase keywords still match at boundaries
    assert _keyword_matches("cho anh 1 cà phê", "cho anh")


def test_address_question_not_routed_to_order():
    # Regression: "add" ⊂ "address" used to misroute this FAQ to ORDER.
    match = classify_by_rules("what is your address?")
    assert match.intent == Intent.FAQ


def test_explicit_order():
    match = classify_by_rules("cho anh 1 bạc xỉu size L")
    assert match.intent == Intent.ORDER


def test_faq_wifi():
    match = classify_by_rules("wifi quán là gì?")
    assert match.intent == Intent.FAQ


def test_consultant_recommendation():
    match = classify_by_rules("có gì ngon rẻ không?")
    assert match.intent == Intent.CONSULTANT
