import pytest
from sqlalchemy.exc import IntegrityError

from app.db.models.catalog import Product, ProductVariant

pytestmark = pytest.mark.integration


def _product() -> Product:
    return Product(
        id="bac-xiu",
        name_vi="Bạc xỉu",
        name_en="Bac xiu",
        category="coffee",
        description="d",
        tags=[],
        ingredients=[],
    )


async def test_variant_price_must_be_positive(db_session):
    db_session.add(_product())
    db_session.add(ProductVariant(id="menu_0001", product_id="bac-xiu", size="M", price_vnd=0))
    with pytest.raises(IntegrityError):
        await db_session.flush()


async def test_variant_size_unique_per_product(db_session):
    db_session.add(_product())
    db_session.add(ProductVariant(id="menu_0001", product_id="bac-xiu", size="M", price_vnd=29000))
    db_session.add(ProductVariant(id="menu_0002", product_id="bac-xiu", size="M", price_vnd=35000))
    with pytest.raises(IntegrityError):
        await db_session.flush()
