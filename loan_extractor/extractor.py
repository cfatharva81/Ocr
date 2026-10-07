"""The extraction pipeline, driven by the YAML field map.

    PDF -> pages (pdf_reader) -> layout + sections (layout) -> fields (this file,
    checkboxes, tables) -> Pydantic validation (schemas) -> consistency checks
    -> confidence -> JSON

Every field is extracted the same way, in this order:
    1. AcroForm widget (if `widget_name` is configured and present)
    2. label-anchored region: right of / below the printed label, checkbox group, ...
    3. regex over the section's text (`regex` in the config)
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import yaml
from pydantic import BaseModel, ValidationError, create_model

from . import __version__
from . import checkboxes as cb
from . import tables
from .layout import LabelMatch, PageLayout, Rect, SectionRegion, ValueMatch, find_sections, union
from .normalizers import currency, mask_sensitive
from .pdf_reader import Page, load_pdf, ocr_region
from .schemas import (GROUP_MODELS, ROW_MODELS, AssetsInformation, Extracted, LiabilitySummary,
                      LoanApplication, OtherAssets, required_fields)

log = logging.getLogger(__name__)

DEFAULT_CONFIG = Path(__file__).parent / "urla_1003.yaml"
NATIVE_CONF = 0.95        # confidence of a value read from the PDF text layer
WIDGET_CONF = 1.0         # ... from an AcroForm field

# Output key order: the core keys first, then the extra groups.
OUTPUT_ORDER = ["borrower_information", "contact_information", "address_information",
                "employment_information", "income_information", "assets", "liabilities",
                "loan_information", "gift_information", "real_estate_information",
                "property_details", "other_assets", "liability_summary"]
INCOME_PARTS = ["base_income", "overtime_income", "bonus_income", "commission_income",
                "military_income", "other_income"]


# ====================================================================== config

def load_config(path=None) -> dict:
    """Load a form field map (YAML). A new form variant = a new YAML file."""
    p = Path(path) if path else DEFAULT_CONFIG
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    for key in ("form_sections", "sections", "fields"):
        if key not in cfg:
            raise ValueError(f"config {p} is missing '{key}'")
    for name, f in cfg["fields"].items():
        if f.get("section") not in cfg["sections"]:
            raise ValueError(f"field {name}: unknown section {f.get('section')!r}")
    cfg["_path"] = str(p)
    return cfg


def all_labels(cfg: dict) -> list[str]:
    """Every printed label and option in the config; a value never runs into one of these."""
    out = list(cfg.get("stop_labels") or [])
    for f in cfg["fields"].values():
        lab = f.get("label")
        out.extend(lab if isinstance(lab, list) else [lab] if lab else [])
        out.extend(tables.options_map(f.get("options") or []).keys())
    return out


# ====================================================================== field results

@dataclass
class FieldResult:
    name: str
    group: str
    value: Any = None                  # raw value handed to the schema
    normalized: Any = None             # value after schema validation
    source_text: str | None = None
    page: int | None = None
    bbox: Rect | None = None
    strategy: str | None = None        # widget | right | below | regex | checkbox:<method> | table:<method> | ...
    source_conf: float = 0.0           # widget 1.0, native 0.95, OCR = mean word confidence
    label_score: float | None = None   # rapidfuzz label score, 0..1
    pattern_ok: bool | None = None
    valid: bool = True
    consistency_ok: bool | None = None
    confidence: float = 0.0
    flags: list[str] = field(default_factory=list)
    checkbox: dict | None = None


def union_rects(rects) -> Rect | None:
    rs = [r for r in rects if r]
    if not rs:
        return None
    return (min(r[0] for r in rs), min(r[1] for r in rs), max(r[2] for r in rs), max(r[3] for r in rs))


@dataclass
class Context:
    cfg: dict
    layouts: list[PageLayout]
    regions: dict[str, SectionRegion]
    warnings: list[str] = field(default_factory=list)

    @property
    def label_threshold(self) -> float:
        return float(self.cfg.get("settings", {}).get("label_threshold", 80))

    @property
    def fill_threshold(self) -> float:
        return float(self.cfg.get("settings", {}).get("checkbox_fill_threshold", 0.15))

    def scope(self, group: str, override: dict | None = None) -> tuple[PageLayout | None, Rect | None]:
        """Page and rectangle of an output group (optionally a field-level scope)."""
        gcfg = self.cfg["sections"][group]
        region = self.regions.get(gcfg["form_section"])
        if region is None:
            return None, None
        lay = self.layouts[region.page_no - 1]
        sc = override or gcfg.get("scope") or {}
        rect, warns = lay.scope_between(region.rect, sc.get("after"), sc.get("before"), self.label_threshold)
        self.warnings.extend(f"{group}: {w}" for w in warns)
        return lay, rect


def _apply_pattern(text: str, pattern: str | None):
    """(value, matched?) - the pattern's group 1 (or whole match) when it matches."""
    if not pattern:
        return text, None
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if not m:
        return text, False
    return (m.group(1) if m.groups() else m.group(0)), True


def _checkbox_conf(lay: PageLayout, decisions, threshold: float) -> float:
    return min((cb.decision_conf(d, lay.page.is_scanned, threshold) for d in decisions), default=0.0)


# ====================================================================== scalar fields

def extract_field(ctx: Context, group: str, name: str, fcfg: dict) -> FieldResult:
    fr = FieldResult(name, group)
    strategy = fcfg.get("strategy", "right")

    # 1. AcroForm widget
    if fcfg.get("widget_name") and strategy in ("right", "below"):
        for lay in ctx.layouts:
            w = next((w for w in lay.page.widgets if w.name == fcfg["widget_name"]), None)
            if w is not None and w.value not in (None, "", "Off"):
                fr.value, fr.pattern_ok = _apply_pattern(str(w.value), fcfg.get("pattern"))
                fr.source_text, fr.page, fr.bbox = str(w.value), lay.page_no, w.rect
                fr.strategy, fr.source_conf = "widget", WIDGET_CONF
                return fr

    lay, scope = ctx.scope(group, fcfg.get("scope"))
    if lay is None:
        fr.flags.append("section_not_found")
        return fr

    # 2. label-anchored
    if strategy in ("right", "below"):
        _label_value(ctx, lay, scope, fcfg, fr)
    elif strategy == "checkbox_group":
        _checkbox_group(ctx, lay, scope, fcfg, fr)
    elif strategy == "checkbox":
        _single_checkbox(ctx, lay, scope, fcfg, fr)
    elif strategy == "count":
        _count(ctx, lay, scope, fcfg, fr)

    # 3. regex over the section text
    if fr.value is None and fcfg.get("regex"):
        _regex(lay, scope, fcfg, fr)
    return fr


def _find_label(ctx: Context, lay: PageLayout, scope: Rect, fcfg: dict) -> LabelMatch | None:
    return lay.find_label(fcfg["label"], scope, ctx.label_threshold,
                          phrase_start=fcfg.get("phrase_start", True), pick="first")


def _label_value(ctx, lay: PageLayout, scope: Rect, fcfg: dict, fr: FieldResult):
    lm = _find_label(ctx, lay, scope, fcfg)
    if lm is None:
        fr.flags.append("label_not_found")
        return
    region = fcfg.get("region") or {}
    if fcfg.get("strategy") == "below":
        vm = lay.value_below(lm, region.get("width", 250), region.get("height", 25), scope)
    else:
        vm = lay.value_right_of(lm, region.get("width", 200), scope)
    fr.label_score = lm.score / 100.0
    if vm is None:
        fr.page, fr.bbox = lay.page_no, lm.bbox
        # A short value is easy for page-level OCR to miss entirely; re-read
        # the region the value would occupy before calling the field empty.
        if lay.page.is_scanned and fcfg.get("strategy") != "below":
            rect = (lm.x1 + 1, lm.y0 - 3, lm.x1 + region.get("width", 120), lm.y1 + 3)
            text, conf = ocr_region(lay.page.image, lay.page.to_px(rect))
            text = lay.cut_at_stop_text(text)
            if text:
                fr.value, fr.pattern_ok = _apply_pattern(text, fcfg.get("pattern"))
                fr.source_text, fr.bbox, fr.source_conf = text, rect, conf
                fr.strategy = (fcfg.get("strategy", "right")) + "+reocr"
                fr.flags.append("re_ocr")
                return
        fr.flags.append("empty")
        return
    fr.value, fr.pattern_ok = _apply_pattern(vm.text, fcfg.get("pattern"))
    fr.source_text, fr.page, fr.bbox = vm.text, lay.page_no, vm.bbox
    fr.strategy = fcfg.get("strategy", "right")
    fr.source_conf = vm.conf if lay.page.is_scanned else NATIVE_CONF
    # "JohnMichael Doe" / "789OakAvenue": OCR lost the spaces between words
    glued_words = bool(re.search(r"[a-z][A-Z]", vm.text)) or (
        len(vm.text) >= 10 and " " not in vm.text and not vm.text.isdigit())
    if lay.page.is_scanned and (fr.pattern_ok is False or glued_words):
        _reread(lay, vm, fcfg, fr)


def _reread(lay: PageLayout, vm: ValueMatch, fcfg: dict, fr: FieldResult):
    """A scanned value failed its pattern: OCR just that region again, whitelisted."""
    x0, y0, x1, y1 = vm.bbox
    text, conf = ocr_region(lay.page.image, lay.page.to_px((x0 - 2, y0 - 3, x1 + 4, y1 + 3)), fast=True)
    val, ok = _apply_pattern(text, fcfg.get("pattern"))
    if text and (ok or (ok is None and " " in text)):
        fr.value, fr.source_text, fr.pattern_ok, fr.source_conf = val, text, ok, conf
        fr.strategy += "+reocr"
        fr.flags.append("re_ocr")


def _checkbox_group(ctx, lay: PageLayout, scope: Rect, fcfg: dict, fr: FieldResult):
    opts = tables.options_map(fcfg["options"])
    lm = _find_label(ctx, lay, scope, fcfg) if fcfg.get("label") else None
    # options are searched in a band around the group's label row
    band = scope if lm is None else (scope[0], lm.y0 - 4, scope[2],
                                     min(scope[3], lm.y1 + (fcfg.get("band") or {}).get("height", 6)))
    g = cb.detect_group(lay, list(opts), band, ctx.fill_threshold, fcfg.get("widget_names"),
                        ctx.label_threshold)
    fr.checkbox = {"selected": g.selected, "ambiguous": g.ambiguous, "methods": g.methods,
                   "missing_options": g.missing_options, "decisions": [d.as_dict() for d in g.decisions]}
    fr.page, fr.label_score = lay.page_no, (lm.score / 100.0) if lm else None
    chosen = [d for d in g.decisions if d.checked]
    if chosen:
        fr.value, fr.source_text = opts[chosen[0].option], chosen[0].option
        fr.bbox = union_rects([chosen[0].box_bbox, chosen[0].label_bbox])
        fr.strategy = "checkbox:" + "+".join(sorted({d.method for d in chosen}))
    else:
        fr.bbox, fr.strategy = band, "checkbox:" + "+".join(g.methods or ["none"])
    if g.ambiguous:   # zero or several boxes ticked in a single-choice group
        fr.flags.append(f"checkbox_group_{'none' if not g.selected else 'multiple'}_checked")
        if len(g.selected) > 1:
            fr.value = None
    fr.source_conf = _checkbox_conf(lay, chosen or g.decisions, ctx.fill_threshold)


def _single_checkbox(ctx, lay: PageLayout, scope: Rect, fcfg: dict, fr: FieldResult):
    lm = lay.find_label(fcfg["label"], scope, ctx.label_threshold, phrase_start=False)
    if lm is None:
        fr.flags.append("label_not_found")
        return
    d = cb.detect_option(lay, lm, ctx.fill_threshold, fcfg.get("widget_name"))
    fr.value = (not d.checked) if fcfg.get("invert") else d.checked
    fr.source_text = f"{'[x]' if d.checked else '[ ]'} {lm.text}"
    fr.page, fr.bbox = lay.page_no, union_rects([d.box_bbox, lm.bbox])
    fr.strategy, fr.label_score = f"checkbox:{d.method}", lm.score / 100.0
    fr.checkbox = {"decisions": [d.as_dict()], "methods": [d.method]}
    fr.source_conf = _checkbox_conf(lay, [d], ctx.fill_threshold)


def _count(ctx, lay: PageLayout, scope: Rect, fcfg: dict, fr: FieldResult):
    """How many copies of a label have a filled-in value (e.g. properties listed)."""
    matches = lay.find_labels(fcfg["label"], scope, ctx.label_threshold)
    width = (fcfg.get("region") or {}).get("width", 200)
    filled = [m for m in matches if lay.value_right_of(m, width, scope)]
    fr.value, fr.page, fr.strategy = len(filled), lay.page_no, "count"
    fr.source_text = f"{len(filled)} of {len(matches)} '{fcfg['label']}' blocks filled"
    fr.bbox = union_rects([m.bbox for m in filled])
    fr.source_conf = 0.85 if lay.page.is_scanned else NATIVE_CONF
    fr.label_score = (min(m.score for m in matches) / 100.0) if matches else None


def _regex(lay: PageLayout, scope: Rect, fcfg: dict, fr: FieldResult):
    text, spans = lay.region_text(scope)
    m = re.search(fcfg["regex"], text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return
    g = 1 if m.groups() else 0
    start, end = m.span(g)
    words = [w for s, e, w in spans if s < end and e > start]
    fr.value = fr.source_text = m.group(g)
    fr.page, fr.bbox, fr.strategy, fr.pattern_ok = lay.page_no, union(words), "regex", True
    fr.source_conf = (sum(w.conf for w in words) / len(words)) if words and lay.page.is_scanned else NATIVE_CONF
    fr.flags = [f for f in fr.flags if f not in ("empty", "label_not_found")]


def extract_group(ctx: Context, group: str) -> dict[str, FieldResult]:
    out = {name: extract_field(ctx, group, name, f)
           for name, f in ctx.cfg["fields"].items() if f["section"] == group}
    for fr in out.values():
        log.debug("%s", mask_sensitive(f"{group}.{fr.name}={fr.source_text!r} via {fr.strategy}"))
    # Rule: "I do not own any real estate" ticked means zero properties.
    owns, count = out.get("owns_real_estate"), out.get("property_count")
    if owns and count and owns.value is False:
        count.value, count.strategy, count.source_conf = 0, "derived", owns.source_conf
    return out


# ====================================================================== tables

def extract_table_group(ctx: Context, tname: str):
    """Returns (rows, totals, metadata) for one configured table."""
    tcfg = ctx.cfg["tables"][tname]
    group = tcfg["section"]
    lay, scope = ctx.scope(group)
    res = tables.extract_table(lay, scope, tcfg, ctx.fill_threshold) if lay else None
    if res is None:
        return [], {}, {"method": "none", "rows": 0, "none_checked": None}

    method = f"table:{res.method}"

    def to_field(name: str, cv: tables.CellValue, header_score: float | None) -> FieldResult:
        fr = FieldResult(name, group, value=cv.value, source_text=cv.text, page=lay.page_no, bbox=cv.bbox)
        fr.strategy = method + (f"+checkbox:{cv.checkbox['method']}" if cv.checkbox and cv.checkbox.get("method")
                                else "") + ("+reocr" if cv.reocr else "")
        fr.source_conf = cv.conf if lay.page.is_scanned else (NATIVE_CONF if cv.value is not None else 0.0)
        fr.label_score = header_score / 100.0 if header_score is not None else None
        fr.checkbox = cv.checkbox
        if cv.checkbox and cv.checkbox.get("ambiguous"):
            fr.flags.append("checkbox_group_ambiguous")
        return fr

    rows = [{f: to_field(f, cv, res.header_scores.get(f)) for f, cv in row.items()} for row in res.rows]
    totals = {name: to_field(name, cv, 100.0) for name, cv in res.totals.items()}

    # Liability totals are not always printed: compute them from the rows then.
    if group == "liabilities":
        for total, col in (("total_unpaid_balance", "unpaid_balance"),
                           ("total_monthly_debt_payment", "monthly_payment")):
            vals = [(currency(r[col].value), r[col].source_conf) for r in rows if col in r]
            vals = [(v, c) for v, c in vals if v is not None]
            if total not in totals and vals:
                totals[total] = FieldResult(total, group, value=f"{sum(v for v, _ in vals):.2f}",
                                            strategy="computed",
                                            source_conf=0.9 * min(c for _, c in vals))
    none_checked = res.none_checked.checked if res.none_checked else None
    return rows, totals, {"method": method, "rows": len(rows), "none_checked": none_checked}


# ====================================================================== validation

_PARTIAL: dict[type, type] = {}


def _partial(model: type[BaseModel]) -> type[BaseModel]:
    """The same model with every field optional, so one bad field never hides the rest.
    Field validators are inherited, so normalization is identical."""
    if model not in _PARTIAL:
        fields = {n: (Optional[f.annotation], None) for n, f in model.model_fields.items()}
        _PARTIAL[model] = create_model(f"{model.__name__}Partial", __base__=model, **fields)
    return _PARTIAL[model]


def validate_fields(model: type[BaseModel], frs: dict[str, FieldResult], where: str) -> list[dict]:
    """Normalize raw values through the schema; invalid values are dropped and flagged."""
    errors = []
    data = {k: fr.value for k, fr in frs.items() if k in model.model_fields}
    obj = None
    for _ in range(len(data) + 1):
        try:
            obj = _partial(model).model_validate(data)
            break
        except ValidationError as e:
            bad = {err["loc"][0] for err in e.errors() if err["loc"] and data.get(err["loc"][0]) is not None}
            errors += [{"field": f"{where}.{err['loc'][0]}", "error": err["msg"]}
                       for err in e.errors() if err["loc"] and err["loc"][0] in bad]
            if not bad:
                break
            for k in bad:
                frs[k].valid = False
                frs[k].flags.append("failed_validation")
                data[k] = None
    for k, fr in frs.items():
        if obj is not None and k in model.model_fields:
            fr.normalized = getattr(obj, k)
    errors += [{"field": f"{where}.{k}", "error": "required field missing"}
               for k in required_fields(model) if k in frs and frs[k].normalized is None]
    return errors


# ====================================================================== confidence

def score(fr: FieldResult) -> float:
    """confidence = source x label match x pattern x validation x consistency."""
    if fr.value is None and fr.normalized is None:
        return 0.0
    c = fr.source_conf
    if fr.label_score is not None:
        c *= 0.5 + 0.5 * fr.label_score
    if fr.strategy == "regex":
        c *= 0.85            # fallback strategy
    if fr.pattern_ok is False:
        c *= 0.6
    if not fr.valid:
        c *= 0.4
    if fr.consistency_ok is False:
        c *= 0.6
    if fr.checkbox and fr.checkbox.get("ambiguous"):
        c = min(c, 0.3)
    return round(max(0.0, min(1.0, c)), 4)


def _sum_check(name, parts: list[tuple[str, FieldResult]], total: tuple[str, FieldResult] | None, tol: float):
    """Do the parts add up to the reported total? Marks every field involved."""
    if total is None or total[1].normalized is None or total[1].strategy == "computed":
        return None
    parts = [(p, fr) for p, fr in parts if fr.normalized is not None]
    if not parts:
        return None
    s = round(sum(fr.normalized for _, fr in parts), 2)
    passed = abs(s - total[1].normalized) <= tol
    for _, fr in parts + [total]:
        fr.consistency_ok = passed and fr.consistency_ok is not False
    return {"check": name, "sum_of_parts": s, "reported_total": total[1].normalized, "passed": passed,
            "fields": [p for p, _ in parts + [total]]}


def consistency_checks(groups, rows, totals, tol: float = 1.0) -> list[dict]:
    checks = []
    inc = groups.get("income_information", {})
    if "total_monthly_income" in inc:
        checks.append(_sum_check("income_components_sum_to_total",
                                 [(f"income_information.{k}", inc[k]) for k in INCOME_PARTS if k in inc],
                                 ("income_information.total_monthly_income", inc["total_monthly_income"]), tol))
    for name, table, col, tgroup, tname in [
            ("asset_values_sum_to_total_assets", "assets", "market_value", "assets", "total_assets"),
            ("other_asset_values_sum_to_total", "other_assets", "market_value", "other_assets", "total_other_assets"),
            ("liability_balances_sum_to_total", "liabilities", "unpaid_balance", "liability_summary",
             "total_unpaid_balance"),
            ("liability_payments_sum_to_total", "liabilities", "monthly_payment", "liability_summary",
             "total_monthly_debt_payment")]:
        tot = totals.get(tgroup, {}).get(tname)
        parts = [(f"{table}[{i}].{col}", r[col]) for i, r in enumerate(rows.get(table, [])) if col in r]
        checks.append(_sum_check(name, parts, (f"{tgroup}.{tname}", tot) if tot else None, tol))
    return [c for c in checks if c]


# ====================================================================== result

def _json(v):
    return v.isoformat() if hasattr(v, "isoformat") else v


@dataclass
class Result:
    groups: dict[str, dict[str, FieldResult]]
    rows: dict[str, list[dict[str, FieldResult]]]
    totals: dict[str, dict[str, FieldResult]]
    pages: list[Page]
    regions: dict[str, SectionRegion]
    metadata: dict = field(default_factory=dict)
    low_threshold: float = 0.7

    def iter_fields(self):
        """(dotted path, FieldResult) for every extracted value."""
        for g, frs in self.groups.items():
            for k, fr in frs.items():
                yield f"{g}.{k}", fr
        for t, rs in self.rows.items():
            for i, r in enumerate(rs):
                for k, fr in r.items():
                    yield f"{t}[{i}].{k}", fr
        for g, frs in self.totals.items():
            for k, fr in frs.items():
                yield f"{g}.{k}", fr

    @property
    def missing_fields(self) -> list[str]:
        return [p for p, fr in self.iter_fields() if fr.normalized is None]

    @property
    def low_confidence_fields(self) -> list[str]:
        return [p for p, fr in self.iter_fields()
                if fr.normalized is not None and fr.confidence < self.low_threshold]

    def data(self, with_metadata: bool = False) -> dict:
        """The output JSON; with_metadata turns every value into an Extracted record."""
        def leaf(fr: FieldResult | None):
            if fr is None:
                return None
            if not with_metadata:
                return _json(fr.normalized)
            return Extracted[Any](value=_json(fr.normalized), confidence=fr.confidence, page=fr.page,
                                  source_text=fr.source_text,
                                  bbox=tuple(round(v, 2) for v in fr.bbox) if fr.bbox else None,
                                  strategy_used=fr.strategy).model_dump(mode="json")

        def items(key):
            return [{k: leaf(fr) for k, fr in r.items()} for r in self.rows.get(key, [])]

        out = {}
        for key in OUTPUT_ORDER:
            if key == "assets":
                out[key] = {"accounts": items(key), "total_assets": leaf(self.totals.get("assets", {}).get("total_assets"))}
            elif key == "other_assets":
                out[key] = {"other_assets_present": leaf(self.groups.get(key, {}).get("other_assets_present")),
                            "other_assets": items(key)}
            elif key in ("liabilities", "gift_information"):
                out[key] = items(key)
            elif key == "liability_summary":
                t = self.totals.get(key, {})
                out[key] = {k: leaf(t.get(k)) for k in ("total_unpaid_balance", "total_monthly_debt_payment")}
            else:
                out[key] = {k: leaf(fr) for k, fr in self.groups.get(key, {}).items()}
        return out

    def to_dict(self, with_metadata: bool = False) -> dict:
        out = self.data(with_metadata)
        out["missing_fields"] = self.missing_fields
        out["low_confidence_fields"] = self.low_confidence_fields
        out["extraction_metadata"] = self.metadata
        return out


# ====================================================================== pipeline

def extract(pdf_path, config=None, force_ocr: bool = False, debug_dir=None) -> Result:
    t0 = time.perf_counter()
    cfg = load_config(config)
    settings = cfg.get("settings") or {}
    low = float(settings.get("low_confidence_threshold", 0.7))

    # 1. read pages, index layout, find sections
    expected = all_labels(cfg) + [a for sec in cfg["form_sections"].values()
                                  for a in sec.get("anchors", [])]
    pages = load_pdf(pdf_path, max_pages=int(cfg.get("max_pages", 3)), force_ocr=force_ocr,
                     expected_text=expected)
    layouts = [PageLayout(p, all_labels(cfg)) for p in pages]
    regions, warnings = find_sections(layouts, cfg["form_sections"], float(settings.get("label_threshold", 80)))
    ctx = Context(cfg, layouts, regions)

    # 2. scalar fields, group by group
    groups = {g: extract_group(ctx, g) for g in cfg["sections"]
              if any(f["section"] == g for f in cfg["fields"].values())}

    # 3. tables
    rows, totals, table_meta = {}, {}, {}
    for tname, tcfg in (cfg.get("tables") or {}).items():
        group = tcfg["section"]
        rows[group], tot, table_meta[tname] = extract_table_group(ctx, tname)
        for name, fr in tot.items():
            tgroup = "liability_summary" if group == "liabilities" else group
            fr.group = tgroup
            totals.setdefault(tgroup, {})[name] = fr
        if group == "other_assets":   # "Check if you don't have any other assets" box
            none_checked = table_meta[tname]["none_checked"]
            present = FieldResult("other_assets_present", group, value=bool(rows[group]) or none_checked is False,
                                  strategy="checkbox" if none_checked is not None else "derived",
                                  source_conf=0.9 if none_checked is not None else 0.8)
            groups["other_assets"] = {"other_assets_present": present}

    # 4. validate + normalize through the Pydantic models
    errors = []
    for g, frs in groups.items():
        model = GROUP_MODELS.get(g) or (OtherAssets if g == "other_assets" else None)
        if model:
            errors += validate_fields(model, frs, g)
    for g, rs in rows.items():
        for i, r in enumerate(rs):
            errors += validate_fields(ROW_MODELS[g], r, f"{g}[{i}]")
    for g, frs in totals.items():
        if g in ("liability_summary", "assets"):
            errors += validate_fields(LiabilitySummary if g == "liability_summary" else AssetsInformation, frs, g)
        else:
            for fr in frs.values():
                fr.normalized = currency(fr.value)

    # 5. consistency checks, then confidence
    checks = consistency_checks(groups, rows, totals, float(settings.get("consistency_tolerance", 1.0)))
    result = Result(groups, rows, totals, pages, regions, low_threshold=low)
    for _, fr in result.iter_fields():
        fr.confidence = score(fr)

    try:
        LoanApplication.model_validate(result.data())
        schema_errors = []
    except ValidationError as e:
        schema_errors = [{"field": ".".join(map(str, err["loc"])), "error": err["msg"]} for err in e.errors()]

    result.metadata = {
        "extractor_version": __version__,
        "form": cfg.get("form"),
        "source_file": Path(pdf_path).name,
        "processing_time_s": None,
        "ocr_engine": "rapidocr" if any(p.is_scanned for p in pages) else None,
        "pages": [{"page_no": p.page_no, "page_type": p.page_type, "file_rotation": p.file_rotation,
                   "rotation_applied": p.rotation_applied, "deskew_angle": p.deskew_angle,
                   "ocr_mean_confidence": p.ocr_mean_conf, "word_count": len(p.words),
                   "widgets": len(p.widgets),
                   "sections": sorted(r.name for r in regions.values() if r.page_no == p.page_no)}
                  for p in pages],
        "sections": {n: {"page": r.page_no, "score": round(r.score, 1), "method": r.method}
                     for n, r in regions.items()},
        "tables": table_meta,
        "checkbox_decisions": [{"field": path, "value": _json(fr.normalized),
                                "decided_by": fr.strategy.split("checkbox:")[-1] if "checkbox:" in (fr.strategy or "")
                                else fr.strategy, **fr.checkbox}
                               for path, fr in result.iter_fields() if fr.checkbox],
        "consistency_checks": checks,
        "validation_errors": errors,
        "schema_valid": not schema_errors,
        "field_flags": {p: fr.flags for p, fr in result.iter_fields() if fr.flags},
        "warnings": list(dict.fromkeys(warnings + ctx.warnings)),
    }
    if schema_errors:
        result.metadata["schema_errors"] = schema_errors
    if debug_dir:
        result.metadata["debug_images"] = [str(p) for p in draw_debug(result, debug_dir)]
    result.metadata["processing_time_s"] = round(time.perf_counter() - t0, 3)
    log.info("extracted %s in %.2fs: %d missing, %d low-confidence, schema_valid=%s", Path(pdf_path).name,
             result.metadata["processing_time_s"], len(result.missing_fields),
             len(result.low_confidence_fields), not schema_errors)
    return result


# ====================================================================== debug images

def draw_debug(result: Result, out_dir) -> list[Path]:
    """Page images with every extracted field's box (green >= 0.9, orange >= low, red below)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for page in result.pages:
        img, s = page.image.copy(), page.scale / 2.8
        for r in result.regions.values():
            if r.page_no == page.page_no:
                x0, y0, x1, y1 = page.to_px(r.rect)
                cv2.rectangle(img, (x0 + 2, y0 + 2), (x1 - 2, y1 - 2), (200, 120, 0), 2)
        for path, fr in result.iter_fields():
            if fr.page != page.page_no or fr.bbox is None:
                continue
            c = fr.confidence
            color = (40, 160, 40) if c >= 0.9 else (0, 140, 255) if c >= result.low_threshold else (40, 40, 220)
            x0, y0, x1, y1 = page.to_px(fr.bbox)
            cv2.rectangle(img, (x0 - 2, y0 - 2), (x1 + 2, y1 + 2), color, 2)
            cv2.putText(img, f"{path.split('.')[-1]} {c:.2f}", (x0, max(10, y0 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4 * s, color, 1, cv2.LINE_AA)
        p = out / f"page_{page.page_no}.png"
        cv2.imwrite(str(p), img)
        paths.append(p)
    return paths


# ====================================================================== logging

class MaskingFilter(logging.Filter):
    """Masks SSNs and account numbers in every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg, record.args = mask_sensitive(record.getMessage()), None
        return True


def setup_logging(level=logging.INFO) -> None:
    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"))
        root.addHandler(h)
    for h in root.handlers:
        if not any(isinstance(f, MaskingFilter) for f in h.filters):
            h.addFilter(MaskingFilter())
