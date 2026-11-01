"""catalog schema

Revision ID: 0001
Revises:
Create Date: 2026-11-01
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "products",
        sa.Column("id", sa.String(64), nullable=False),
        sa.Column("name_vi", sa.String(200), nullable=False),
        sa.Column("name_en", sa.String(200), nullable=False),
        sa.Column("category", sa.String(16), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("tags", postgresql.ARRAY(sa.Text()), server_default="{}", nullable=False),
        sa.Column("ingredients", postgresql.ARRAY(sa.Text()), server_default="{}", nullable=False),
        sa.Column("image_url", sa.Text(), nullable=True),
        sa.Column("status", sa.String(16), server_default="active", nullable=False),
        sa.Column("sort_order", sa.Integer(), server_default="0", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("category IN ('coffee','tea','freeze','food')", name="ck_products_category"),
        sa.CheckConstraint("status IN ('active','sold_out','hidden')", name="ck_products_status"),
        sa.PrimaryKeyConstraint("id", name="pk_products"),
    )
    op.create_table(
        "product_variants",
        sa.Column("id", sa.String(32), nullable=False),
        sa.Column("product_id", sa.String(64), nullable=False),
        sa.Column("size", sa.String(1), nullable=False),
        sa.Column("price_vnd", sa.Integer(), nullable=False),
        sa.CheckConstraint("price_vnd > 0", name="ck_product_variants_price_positive"),
        sa.CheckConstraint("size IN ('S','M','L')", name="ck_product_variants_size"),
        sa.ForeignKeyConstraint(
            ["product_id"], ["products.id"],
            name="fk_product_variants_product_id_products", ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_product_variants"),
        sa.UniqueConstraint("product_id", "size", name="uq_product_variants_product_id_size"),
    )
    op.create_table(
        "option_groups",
        sa.Column("code", sa.String(32), nullable=False),
        sa.Column("name_vi", sa.String(100), nullable=False),
        sa.Column("choices", postgresql.JSONB(), nullable=False),
        sa.Column("default_choice", sa.String(32), nullable=False),
        sa.PrimaryKeyConstraint("code", name="pk_option_groups"),
    )
    op.create_table(
        "product_option_groups",
        sa.Column("product_id", sa.String(64), nullable=False),
        sa.Column("group_code", sa.String(32), nullable=False),
        sa.ForeignKeyConstraint(
            ["product_id"], ["products.id"],
            name="fk_product_option_groups_product_id_products", ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["group_code"], ["option_groups.code"],
            name="fk_product_option_groups_group_code_option_groups", ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("product_id", "group_code", name="pk_product_option_groups"),
    )
    op.create_table(
        "faqs",
        sa.Column("id", sa.String(32), nullable=False),
        sa.Column("topic", sa.String(100), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("language", sa.String(8), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_faqs"),
    )
    op.create_table(
        "policy_chunks",
        sa.Column("id", sa.String(32), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("language", sa.String(8), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("entities", postgresql.JSONB(), server_default="[]", nullable=False),
        sa.Column("source_file", sa.String(255), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_policy_chunks"),
    )
    op.create_table(
        "shop_settings",
        sa.Column("key", sa.String(100), nullable=False),
        sa.Column("value", postgresql.JSONB(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("key", name="pk_shop_settings"),
    )


def downgrade() -> None:
    for table in (
        "shop_settings",
        "policy_chunks",
        "faqs",
        "product_option_groups",
        "option_groups",
        "product_variants",
        "products",
    ):
        op.drop_table(table)
