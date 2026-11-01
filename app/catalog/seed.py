import csv
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import delete, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.catalog import (
    Faq,
    OptionGroup,
    PolicyChunk,
    Product,
    ProductOptionGroup,
    ProductVariant,
    ShopSetting,
)

_SIZE_CLAUSE = re.compile(r"\s*size\s+[SML]\s*\([^)]*\)")
_SIZE_ORDER = {"S": 0, "M": 1, "L": 2}

OPTION_GROUPS = [
    {
        "code": "sugar",
        "name_vi": "Đường",
        "choices": [
            {"code": "khong", "label_vi": "Không đường"},
            {"code": "it", "label_vi": "Ít đường"},
            {"code": "binh_thuong", "label_vi": "Bình thường"},
            {"code": "nhieu", "label_vi": "Nhiều đường"},
        ],
        "default_choice": "binh_thuong",
    },
    {
        "code": "ice",
        "name_vi": "Đá",
        "choices": [
            {"code": "khong", "label_vi": "Không đá"},
            {"code": "it", "label_vi": "Ít đá"},
            {"code": "binh_thuong", "label_vi": "Bình thường"},
        ],
        "default_choice": "binh_thuong",
    },
    {
        "code": "temperature",
        "name_vi": "Nóng/Đá",
        "choices": [
            {"code": "da", "label_vi": "Đá"},
            {"code": "nong", "label_vi": "Nóng"},
        ],
        "default_choice": "da",
    },
]

SHOP_SETTINGS = {
    "delivery": {"fee_vnd": 15000, "max_days_ahead": 2, "open": "07:00", "close": "22:00"},
    "queue": {"scheduled_lead_minutes": 20},
}


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def slugify(name: str) -> str:
    s = nfc(name).replace("đ", "d").replace("Đ", "D")
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")


@dataclass
class VariantSeed:
    id: str
    size: str
    price_vnd: int


@dataclass
class ProductSeed:
    id: str
    name_vi: str
    name_en: str
    category: str
    description: str
    tags: list[str]
    ingredients: list[str]
    status: str
    sort_order: int
    variants: list[VariantSeed] = field(default_factory=list)
    option_groups: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SeedStats:
    products: int
    variants: int
    faqs: int
    chunks: int


def option_groups_for(category: str, tags: set[str]) -> list[str]:
    if category == "food":
        return []
    if "hot" in tags:
        return ["sugar"]
    if "iced" in tags:
        return ["sugar", "ice"]
    if category == "freeze":
        return ["sugar"]
    if category == "tea":
        return ["sugar", "ice"]
    if category == "coffee":
        return ["sugar", "ice", "temperature"]
    return []


def _split(value: str) -> list[str]:
    return [nfc(p.strip()) for p in value.split("|") if p.strip()]


def group_menu_rows(rows: list[dict]) -> list[ProductSeed]:
    products: dict[str, ProductSeed] = {}
    for row in rows:
        name_vi = nfc(row["name_vi"])
        pid = slugify(name_vi)
        tags = _split(row["tags"])
        ingredients = _split(row["ingredients"])
        available = str(row["available"]).strip().lower() == "true"
        p = products.get(pid)
        if p is None:
            p = products[pid] = ProductSeed(
                id=pid,
                name_vi=name_vi,
                name_en=nfc(row["name_en"]),
                category=row["category"],
                description="",
                tags=[],
                ingredients=[],
                status="active" if available else "sold_out",
                sort_order=len(products),
            )
        if row["size"] == "S" or not p.description:
            p.description = nfc(_SIZE_CLAUSE.sub("", row["description"])).strip()
        p.tags += [t for t in tags if t not in p.tags]
        p.ingredients += [i for i in ingredients if i not in p.ingredients]
        p.variants.append(VariantSeed(row["id"], row["size"], int(row["price"])))
    for p in products.values():
        p.variants.sort(key=lambda v: _SIZE_ORDER[v.size])
        p.option_groups = option_groups_for(p.category, set(p.tags))
    return list(products.values())


def _upsert(db: AsyncSession, model, rows: list[dict], key: list[str]):
    stmt = insert(model).values(rows)
    update = {c: stmt.excluded[c] for c in rows[0] if c not in key}
    return db.execute(stmt.on_conflict_do_update(index_elements=key, set_=update))


async def seed_catalog(db: AsyncSession, data_dir: Path, *, reset: bool = False) -> SeedStats:
    with open(data_dir / "menu.csv", encoding="utf-8", newline="") as f:
        products = group_menu_rows(list(csv.DictReader(f)))
    with open(data_dir / "faq.csv", encoding="utf-8", newline="") as f:
        faqs = [
            {
                "id": r["id"],
                "topic": nfc(r["topic"]),
                "question": nfc(r["question"]),
                "answer": nfc(r["answer"]),
                "language": r["language"],
            }
            for r in csv.DictReader(f)
        ]
    chunks = []
    with open(data_dir / "docs.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            chunks.append(
                {
                    "id": d["id"],
                    "chunk_index": d["chunk_index"],
                    "language": d["language"],
                    "text": nfc(d["text"]),
                    "entities": d.get("entities", []),
                    "source_file": d["source_file"],
                }
            )

    if reset:
        await db.execute(
            text(
                "TRUNCATE products, option_groups, faqs, policy_chunks, shop_settings CASCADE"
            )
        )

    await _upsert(db, OptionGroup, OPTION_GROUPS, ["code"])
    await _upsert(
        db,
        Product,
        [
            {
                "id": p.id,
                "name_vi": p.name_vi,
                "name_en": p.name_en,
                "category": p.category,
                "description": p.description,
                "tags": p.tags,
                "ingredients": p.ingredients,
                "status": p.status,
                "sort_order": p.sort_order,
            }
            for p in products
        ],
        ["id"],
    )
    await _upsert(
        db,
        ProductVariant,
        [
            {"id": v.id, "product_id": p.id, "size": v.size, "price_vnd": v.price_vnd}
            for p in products
            for v in p.variants
        ],
        ["id"],
    )
    await db.execute(
        delete(ProductOptionGroup).where(ProductOptionGroup.product_id.in_([p.id for p in products]))
    )
    links = [
        {"product_id": p.id, "group_code": g} for p in products for g in p.option_groups
    ]
    await db.execute(insert(ProductOptionGroup).values(links))
    await _upsert(db, Faq, faqs, ["id"])
    await _upsert(db, PolicyChunk, chunks, ["id"])
    await _upsert(
        db,
        ShopSetting,
        [{"key": k, "value": v} for k, v in SHOP_SETTINGS.items()],
        ["key"],
    )
    await db.commit()
    return SeedStats(
        products=len(products),
        variants=sum(len(p.variants) for p in products),
        faqs=len(faqs),
        chunks=len(chunks),
    )
