from typing import Any

from pydantic import BaseModel


class VariantDTO(BaseModel):
    id: str
    size: str
    price_vnd: int


class ProductDTO(BaseModel):
    id: str
    name_vi: str
    name_en: str
    category: str
    description: str
    tags: list[str]
    ingredients: list[str]
    image_url: str | None
    status: str
    variants: list[VariantDTO]
    option_groups: list[str]


class FaqDTO(BaseModel):
    id: str
    topic: str
    question: str
    answer: str
    language: str


class PolicyChunkDTO(BaseModel):
    id: str
    chunk_index: int
    language: str
    text: str
    entities: list[dict[str, Any]]
    source_file: str


class OptionChoiceDTO(BaseModel):
    code: str
    label_vi: str


class OptionGroupDTO(BaseModel):
    code: str
    name_vi: str
    choices: list[OptionChoiceDTO]
    default_choice: str
