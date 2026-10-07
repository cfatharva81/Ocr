"""A schema for any document: the LLM names the sections and fields, Pydantic checks the shape.

Gemini's structured output needs a fixed schema, so the schema is generic instead of
form-specific: a document is sections of key/label/value fields plus tables. The model
picks the keys and a value_type; `normalized` then turns the printed value into a typed
one with the same normalizers the rule-based extractor uses.

Auto mode's keys can change from run to run. DynamicDocument.to_template() freezes one
result's structure (keys, labels, types; no values) as a Template; template_model() turns
that into a Pydantic model with exactly those keys, so later runs fill the same structure.
"""
from __future__ import annotations

import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, computed_field, create_model, field_validator, model_validator

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

    def to_template(self) -> "Template":
        """This result's structure without its values; keys match to_nested()."""
        taken, sections = {"document_type"}, []
        for s in self.sections:
            names: set[str] = set()
            fields = [TemplateField(key=fresh(names, f.key), label=f.label, value_type=f.value_type)
                      for f in s.fields]
            tables = [TemplateTable(key=fresh(names, t.key), title=t.title,
                                    columns=[TemplateColumn(key=c, value_type=ct)
                                             for c, ct in zip(t.columns, t.column_types)])
                      for t in s.tables]
            sections.append(TemplateSection(key=fresh(taken, s.key), title=s.title, fields=fields, tables=tables))
        return Template(document_type=self.document_type, sections=sections)


# ====================================================================== templates
# A template is plain JSON, so it can be reviewed and edited by hand: rename a key,
# drop a field, fix a value_type.

class TemplateField(_Base):
    key: str
    label: str
    value_type: ValueType = "text"


class TemplateColumn(_Base):
    key: str
    value_type: ValueType = "text"


class TemplateTable(_Base):
    key: str
    title: Optional[str] = None
    columns: list[TemplateColumn]


class TemplateSection(_Base):
    key: str
    title: Optional[str] = None
    fields: list[TemplateField] = Field(default_factory=list)
    tables: list[TemplateTable] = Field(default_factory=list)


class Template(_Base):
    template_version: int = 1
    document_type: str
    sections: list[TemplateSection]

    @model_validator(mode="after")
    def _unique_keys(self):
        """Keys must be snake_case and unique where they share a JSON object."""
        def check(keys: list[str], where: str):
            bad = [k for k in keys if snake(k) != k]
            if bad:
                raise ValueError(f"{where}: keys must be snake_case: {bad}")
            dup = {k for k in keys if keys.count(k) > 1}
            if dup:
                raise ValueError(f"{where}: duplicate keys {sorted(dup)}")
        check(["document_type"] + [s.key for s in self.sections], "sections")
        for s in self.sections:
            check([f.key for f in s.fields] + [t.key for t in s.tables], s.key)
            for t in s.tables:
                check([c.key for c in t.columns], f"{s.key}.{t.key}")
        return self


_FILL_CONFIG = ConfigDict(extra="forbid", str_strip_whitespace=True, populate_by_name=True)


def template_model(tpl: Template) -> type[BaseModel]:
    """A Pydantic model with exactly the template's keys; every value is the printed text or null.

    Keys become aliases (python names f0, f1 ...), so any key works, even "copy" or "1a".
    Each field's description carries its printed label and value type for the LLM.
    """
    def model(name: str, items: list[tuple[str, Any, str]]) -> type[BaseModel]:
        return create_model(name, __config__=_FILL_CONFIG, **{
            f"f{i}": (ann, Field(..., alias=key, description=desc)) for i, (key, ann, desc) in enumerate(items)})

    sections = []
    for si, s in enumerate(tpl.sections):
        items: list[tuple[str, Any, str]] = [
            (f.key, Optional[str], f'printed label "{f.label}"; {f.value_type}') for f in s.fields]
        for ti, t in enumerate(s.tables):
            row = model(f"S{si}T{ti}_{t.key}", [(c.key, Optional[str], c.value_type) for c in t.columns])
            title = f' "{t.title}"' if t.title else ""
            items.append((t.key, list[row], f"table{title}: one item per filled-in row"))
        desc = f'section "{s.title}"' if s.title else f"section {s.key}"
        sections.append((s.key, model(f"S{si}_{s.key}", items), desc))
    return model("FilledTemplate", sections)


def fill_template(tpl: Template, filled: BaseModel) -> dict:
    """The filled model as {document_type, <section>: {<field>: value, <table>: [rows]}}, values typed."""
    data = filled.model_dump(by_alias=True)
    out: dict = {"document_type": tpl.document_type}
    for s in tpl.sections:
        sec = data[s.key]
        out[s.key] = {f.key: normalize(sec[f.key], f.value_type) for f in s.fields}
        for t in s.tables:
            out[s.key][t.key] = [{c.key: normalize(r[c.key], c.value_type) for c in t.columns} for r in sec[t.key]]
    return out
