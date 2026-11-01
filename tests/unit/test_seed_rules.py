import csv
import json
import unicodedata
from pathlib import Path

from app.catalog.seed import group_menu_rows, option_groups_for, slugify

DATA = Path(__file__).resolve().parents[2] / "data" / "mock"


def _rows() -> list[dict]:
    with open(DATA / "menu.csv", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_option_groups_rules():
    assert option_groups_for("coffee", {"coffee", "milk", "iced", "popular"}) == ["sugar", "ice"]
    assert option_groups_for("coffee", {"coffee", "black", "hot"}) == ["sugar"]
    assert option_groups_for("coffee", set()) == ["sugar", "ice", "temperature"]
    assert option_groups_for("freeze", {"chocolate"}) == ["sugar"]
    assert option_groups_for("tea", {"fruity"}) == ["sugar", "ice"]
    assert option_groups_for("food", {"iced"}) == []


def test_group_menu_rows_counts():
    products = group_menu_rows(_rows())
    assert len(products) == 21
    assert sum(len(p.variants) for p in products) == 55
    bx = next(p for p in products if p.id == "bac-xiu")
    assert [v.size for v in bx.variants] == ["S", "M", "L"]
    assert [v.price_vnd for v in bx.variants] == [35000, 45000, 55000]
    assert "size" not in bx.description and "đ)" not in bx.description
    by_id = {p.id: p for p in products}
    assert by_id["latte"].option_groups == ["sugar", "ice", "temperature"]
    assert by_id["sung-bo"].option_groups == []


def test_slug():
    assert slugify("Bạc xỉu") == "bac-xiu"
    assert slugify("Trà đào cam sả") == "tra-dao-cam-sa"


def test_nfc_normalization():
    rows = [dict(r) for r in _rows() if r["name_vi"] == "Trà vải"]
    for r in rows:
        r["name_vi"] = unicodedata.normalize("NFD", r["name_vi"])
    products = group_menu_rows(rows)
    assert products[0].name_vi == unicodedata.normalize("NFC", "Trà vải")
    assert products[0].name_vi != unicodedata.normalize("NFD", "Trà vải")


def test_no_third_party_delivery_claims():
    with open(DATA / "faq.csv", encoding="utf-8", newline="") as f:
        faqs = list(csv.DictReader(f))
    for faq in faqs:
        if "GrabFood" in faq["answer"]:
            assert faq["id"] == "faq_0011"
            assert "không bán qua GrabFood" in faq["answer"]
        assert "không nhận đặt bàn trước" not in faq["answer"]
    with open(DATA / "docs.jsonl", encoding="utf-8") as f:
        for line in f:
            text = json.loads(line)["text"]
            assert "GrabFood" not in text
            assert "không nhận đặt bàn trước" not in text
