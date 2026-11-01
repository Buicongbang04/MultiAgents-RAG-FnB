from pathlib import Path

import pytest
from sqlalchemy import func, select

from app.catalog.repository import get_product, get_shop_setting, list_option_groups
from app.catalog.seed import SeedStats, seed_catalog
from app.db.models.catalog import Faq, PolicyChunk, Product, ProductVariant

pytestmark = pytest.mark.integration

DATA = Path(__file__).resolve().parents[2] / "data" / "mock"


async def _counts(db):
    return tuple(
        [
            (await db.execute(select(func.count()).select_from(m))).scalar_one()
            for m in (Product, ProductVariant, Faq, PolicyChunk)
        ]
    )


async def test_seed_is_idempotent(db_session):
    first = await seed_catalog(db_session, DATA)
    assert first == SeedStats(21, 55, 30, 20)
    counts = await _counts(db_session)
    second = await seed_catalog(db_session, DATA)
    assert second == SeedStats(21, 55, 30, 20)
    assert await _counts(db_session) == counts == (21, 55, 30, 20)
    assert (await get_shop_setting(db_session, "queue")) == {"scheduled_lead_minutes": 20}
    assert len(await list_option_groups(db_session)) == 3


async def test_reset_reseeds(db_session):
    await seed_catalog(db_session, DATA)
    assert await seed_catalog(db_session, DATA, reset=True) == SeedStats(21, 55, 30, 20)
    assert await _counts(db_session) == (21, 55, 30, 20)


async def test_repository_returns_variants_sorted(db_session):
    await seed_catalog(db_session, DATA)
    p = await get_product(db_session, "bac-xiu")
    assert p is not None
    assert [v.size for v in p.variants] == ["S", "M", "L"]
    assert p.variants[0].id == "menu_0003"
    assert p.option_groups == ["sugar", "ice"]
    assert await get_product(db_session, "nope") is None
