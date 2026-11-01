from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.catalog.schemas import (
    FaqDTO,
    OptionGroupDTO,
    PolicyChunkDTO,
    ProductDTO,
    VariantDTO,
)
from app.db.models.catalog import (
    Faq,
    OptionGroup,
    PolicyChunk,
    Product,
    ProductOptionGroup,
    ProductStatus,
    ProductVariant,
    ShopSetting,
)

_SIZE_ORDER = {"S": 0, "M": 1, "L": 2}


async def _to_dtos(db: AsyncSession, products: list[Product]) -> list[ProductDTO]:
    ids = [p.id for p in products]
    variants: dict[str, list[ProductVariant]] = {i: [] for i in ids}
    groups: dict[str, list[str]] = {i: [] for i in ids}
    if ids:
        for v in (
            await db.execute(select(ProductVariant).where(ProductVariant.product_id.in_(ids)))
        ).scalars():
            variants[v.product_id].append(v)
        # option_groups ordered sugar -> ice -> temperature (canonical display order)
        order = {"sugar": 0, "ice": 1, "temperature": 2}
        for link in (
            await db.execute(
                select(ProductOptionGroup).where(ProductOptionGroup.product_id.in_(ids))
            )
        ).scalars():
            groups[link.product_id].append(link.group_code)
        for g in groups.values():
            g.sort(key=lambda c: order.get(c, 99))
    return [
        ProductDTO(
            id=p.id,
            name_vi=p.name_vi,
            name_en=p.name_en,
            category=p.category,
            description=p.description,
            tags=list(p.tags),
            ingredients=list(p.ingredients),
            image_url=p.image_url,
            status=ProductStatus(p.status).value,
            variants=[
                VariantDTO(id=v.id, size=v.size, price_vnd=v.price_vnd)
                for v in sorted(variants[p.id], key=lambda v: _SIZE_ORDER[v.size])
            ],
            option_groups=groups[p.id],
        )
        for p in products
    ]


async def list_products(db: AsyncSession, *, include_hidden: bool = False) -> list[ProductDTO]:
    stmt = select(Product).order_by(Product.sort_order, Product.id)
    if not include_hidden:
        stmt = stmt.where(Product.status != ProductStatus.hidden)
    return await _to_dtos(db, list((await db.execute(stmt)).scalars()))


async def get_product(db: AsyncSession, product_id: str) -> ProductDTO | None:
    p = await db.get(Product, product_id)
    return (await _to_dtos(db, [p]))[0] if p else None


async def list_faqs(db: AsyncSession) -> list[FaqDTO]:
    rows = (await db.execute(select(Faq).order_by(Faq.id))).scalars()
    return [FaqDTO(id=r.id, topic=r.topic, question=r.question, answer=r.answer, language=r.language) for r in rows]


async def list_policy_chunks(db: AsyncSession) -> list[PolicyChunkDTO]:
    rows = (await db.execute(select(PolicyChunk).order_by(PolicyChunk.id))).scalars()
    return [
        PolicyChunkDTO(
            id=r.id,
            chunk_index=r.chunk_index,
            language=r.language,
            text=r.text,
            entities=list(r.entities),
            source_file=r.source_file,
        )
        for r in rows
    ]


async def list_option_groups(db: AsyncSession) -> list[OptionGroupDTO]:
    rows = (await db.execute(select(OptionGroup))).scalars()
    order = {"sugar": 0, "ice": 1, "temperature": 2}
    return sorted(
        (
            OptionGroupDTO(
                code=r.code, name_vi=r.name_vi, choices=r.choices, default_choice=r.default_choice
            )
            for r in rows
        ),
        key=lambda g: order.get(g.code, 99),
    )


async def get_shop_setting(db: AsyncSession, key: str) -> dict[str, Any] | None:
    row = await db.get(ShopSetting, key)
    return dict(row.value) if row else None
