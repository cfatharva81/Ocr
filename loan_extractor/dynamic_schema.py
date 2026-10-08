"""A schema for any document: the LLM names the sections and fields, Pydantic checks the shape.

Gemini's structured output needs a fixed schema, so the schema is generic instead of
form-specific: a document is sections of key/label/value fields plus tables. The model
picks the keys and a value_type; `normalized` then turns the printed value into a typed
one with shared value normalizers.

The model chooses keys and value types for each document, so the schema can change from
one extraction to the next.
"""
from __future__ import annotations

import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from . import normalizers as N

ValueType = Literal["text", "integer", "number", "currency", "percent", "date", "boolean",
                    "phone", "email", "id"]


def snake(s: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_").lower()
    return s or "field"


def fresh(taken: set[str], key: str) -> str:
    """`key`, or key_2, key_3 ... if it is already taken; records the result."""
    out, i = key, 2
    while out in taken:
        out, i = f"{key}_{i}", i + 1
    taken.add(out)
    return out


def normalize(value: str | None, value_type: str) -> Any:
    """A printed value as a JSON type: '$6,500.00' -> 6500.0, '03/15/1985' -> '1985-03-15'."""
    if value is None or not N.clean_text(value):
        return None
    fn = {"currency": N.currency, "number": N.currency, "integer": N.integer,
          "date": N.iso_date, "boolean": N.boolean, "phone": N.phone, "email": N.email,
          "percent": lambda s: N.currency(s.replace("%", ""))}.get(value_type)
    out = fn(value) if fn else None
    return out if out is not None else N.clean_text(value)


class _Base(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


class DynamicField(_Base):
    key: str = Field(description="snake_case name you choose for this field, e.g. borrower_name, base_income")
    label: str = Field(description="the field's label exactly as printed on the document")
    value: Optional[str] = Field(description="the value exactly as printed; null if the field is blank")
    value_type: ValueType = Field(description="what kind of value this is")

    @field_validator("key")
    @classmethod
    def _snake(cls, v: str) -> str:
        return snake(v)

    @computed_field
    @property
    def normalized(self) -> Any:
        return normalize(self.value, self.value_type)


class DynamicTable(_Base):
    key: str = Field(description="snake_case name you choose for this table, e.g. assets, liabilities")
    title: Optional[str] = Field(description="the table's heading as printed, or null")
    columns: list[str] = Field(description="snake_case column names you choose, in printed order")
    column_types: list[ValueType] = Field(description="the value_type of each column, aligned with columns")
    rows: list[list[Optional[str]]] = Field(
        description="one list per filled-in row, values as printed, aligned with columns; skip empty rows")

    @field_validator("key")
    @classmethod
    def _snake(cls, v: str) -> str:
        return snake(v)

    @field_validator("columns")
    @classmethod
    def _cols(cls, v: list[str]) -> list[str]:
        taken: set[str] = set()
        return [fresh(taken, snake(c)) for c in v]

    @field_validator("column_types", "rows")
    @classmethod
    def _width(cls, v, info):
        """Pad or cut to the number of columns (missing column types become text)."""
        n = len(info.data.get("columns") or [])
        if info.field_name == "column_types":
            return (v + ["text"] * n)[:n]
        return [(r + [None] * n)[:n] for r in v]

    def records(self) -> list[dict]:
        return [{c: normalize(v, t) for c, t, v in zip(self.columns, self.column_types, r)} for r in self.rows]


class DynamicSection(_Base):
    key: str = Field(description="snake_case name you choose for this section, e.g. borrower_information")
    title: Optional[str] = Field(description="the section heading as printed, or null")
    fields: list[DynamicField]
    tables: list[DynamicTable]

    @field_validator("key")
    @classmethod
    def _snake(cls, v: str) -> str:
        return snake(v)


class DynamicDocument(_Base):
    document_type: str = Field(description="what this document is, e.g. 'URLA Form 1003 loan application', 'invoice'")
    sections: list[DynamicSection]

    def to_nested(self) -> dict:
        """{document_type, <section>: {<field>: value, <table>: [{<column>: value}]}}; repeated keys get _2, _3."""
        out: dict = {"document_type": self.document_type}
        taken = {"document_type"}
        for s in self.sections:
            sec: dict = {}
            names: set[str] = set()
            for f in s.fields:
                sec[fresh(names, f.key)] = f.normalized
            for t in s.tables:
                sec[fresh(names, t.key)] = t.records()
            out[fresh(taken, s.key)] = sec
        return out
