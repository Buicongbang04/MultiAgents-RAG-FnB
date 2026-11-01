import unicodedata

from app.rag import menu


def test_get_menu_variants_matches_decomposed_name(monkeypatch):
    nfc = unicodedata.normalize("NFC", "Bạc xỉu")
    nfd = unicodedata.normalize("NFD", "Bạc xỉu")
    assert nfc != nfd

    seen = {}

    def fake_query(query, params=None):
        seen["names"] = params["names"]
        return [
            {"name": nfc, "id": "m-L", "name_vi": nfc, "size": "L", "price": 49000,
             "category": "coffee", "description": None},
            {"name": nfc, "id": "m-S", "name_vi": nfc, "size": "S", "price": 39000,
             "category": "coffee", "description": None},
        ]

    monkeypatch.setattr(menu.neo4j_client, "execute_query", fake_query)

    result = menu.get_menu_variants([nfd])

    assert seen["names"] == [nfc]
    assert list(result) == [nfd]
    assert [v["size"] for v in result[nfd]] == ["S", "L"]
