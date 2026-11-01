import enum
from datetime import datetime
from typing import Any

from sqlalchemy import (
    ARRAY,
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class ProductStatus(str, enum.Enum):
    active = "active"
    sold_out = "sold_out"
    hidden = "hidden"


class Product(Base):
    __tablename__ = "products"
    __table_args__ = (
        CheckConstraint("category IN ('coffee','tea','freeze','food')", name="category"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # slug, e.g. "bac-xiu"
    name_vi: Mapped[str] = mapped_column(String(200))
    name_en: Mapped[str] = mapped_column(String(200))
    category: Mapped[str] = mapped_column(String(16))
    description: Mapped[str] = mapped_column(Text)
    tags: Mapped[list[str]] = mapped_column(ARRAY(Text), server_default="{}")
    ingredients: Mapped[list[str]] = mapped_column(ARRAY(Text), server_default="{}")
    image_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[ProductStatus] = mapped_column(
        Enum(
            ProductStatus,
            native_enum=False,
            length=16,
            create_constraint=True,
            name="status",
            values_callable=lambda e: [m.value for m in e],
        ),
        default=ProductStatus.active,
        server_default=ProductStatus.active.value,
    )
    sort_order: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class ProductVariant(Base):
    __tablename__ = "product_variants"
    __table_args__ = (
        CheckConstraint("price_vnd > 0", name="price_positive"),
        CheckConstraint("size IN ('S','M','L')", name="size"),
        UniqueConstraint("product_id", "size", name="uq_product_variants_product_id_size"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)  # legacy menu_0001
    product_id: Mapped[str] = mapped_column(ForeignKey("products.id", ondelete="CASCADE"))
    size: Mapped[str] = mapped_column(String(1))
    price_vnd: Mapped[int] = mapped_column(Integer)


class OptionGroup(Base):
    __tablename__ = "option_groups"

    code: Mapped[str] = mapped_column(String(32), primary_key=True)  # sugar|ice|temperature
    name_vi: Mapped[str] = mapped_column(String(100))
    choices: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)  # ordered [{"code","label_vi"}]
    default_choice: Mapped[str] = mapped_column(String(32))


class ProductOptionGroup(Base):
    __tablename__ = "product_option_groups"

    product_id: Mapped[str] = mapped_column(
        ForeignKey("products.id", ondelete="CASCADE"), primary_key=True
    )
    group_code: Mapped[str] = mapped_column(
        ForeignKey("option_groups.code", ondelete="CASCADE"), primary_key=True
    )


class Faq(Base):
    __tablename__ = "faqs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)  # legacy faq_0001
    topic: Mapped[str] = mapped_column(String(100))
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    language: Mapped[str] = mapped_column(String(8))


class PolicyChunk(Base):
    __tablename__ = "policy_chunks"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)  # legacy doc_0001
    chunk_index: Mapped[int] = mapped_column(Integer)
    language: Mapped[str] = mapped_column(String(8))
    text: Mapped[str] = mapped_column(Text)
    entities: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, server_default="[]")
    source_file: Mapped[str] = mapped_column(String(255))


class ShopSetting(Base):
    __tablename__ = "shop_settings"

    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[dict[str, Any]] = mapped_column(JSONB)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
