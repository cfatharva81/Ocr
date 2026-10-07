"""Assets / liabilities / gifts tables.

Two methods, tried in order:
  1. pymupdf  - page.find_tables() on native pages (ruled tables)
  2. cluster  - words grouped into rows by y and assigned to columns using the
                x-positions of the header labels (works on scans too)

The header row is found by fuzzy-matching the configured column headers.
Cell text always comes from the page's own words (native or OCR) inside the
cell box. Rows are read until the total row; rows whose key columns are all
empty are skipped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pymupdf as fitz
from rapidfuzz import fuzz

from . import checkboxes as cb
from .layout import PageLayout, Rect, group_rows, join, mean_conf, norm, union
from .pdf_reader import Word, ocr_region

HEADER_MIN = 75.0
MAX_ROW_GAP = 120.0      # points


@dataclass
class Cell:
    bbox: Rect


@dataclass
class Grid:
    rows: list[list[Cell]]
    method: str
    bbox: Rect


@dataclass
class CellValue:
    text: str | None
    value: Any
    bbox: Rect | None
    conf: float
    words: list[Word] = field(default_factory=list)
    checkbox: dict | None = None
    reocr: bool = False


@dataclass
class TableResult:
    rows: list[dict[str, CellValue]]
    totals: dict[str, CellValue]
    method: str
    header_scores: dict[str, float]
    page_no: int
    none_checked: cb.CheckboxDecision | None = None


def header_score(alias: str, text: str) -> float:
    a, t = norm(alias), norm(text)
    if not a or not t:
        return 0.0
    return max(fuzz.ratio(a, t), 0.95 * fuzz.token_set_ratio(a, t))


def _intersects(a: Rect, b: Rect) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


# ------------------------------------------------------------------ backends

def pymupdf_grids(lay: PageLayout, scope: Rect) -> list[Grid]:
    page = lay.page
    if page.is_scanned or not page.source_path:
        return []
    doc = fitz.open(page.source_path)
    try:
        fp = doc[page.page_no - 1]
        fp.set_rotation(0)
        tabs = fp.find_tables().tables
        fp.set_rotation(page.render_rotation)
        m = fp.rotation_matrix
        grids = []
        for t in tabs:
            rows = []
            for r in t.rows:
                cells = []
                for c in r.cells:
                    if c is None:
                        cells.append(Cell((0, 0, 0, 0)))
                    else:
                        rr = fitz.Rect(c) * m
                        cells.append(Cell((rr.x0, rr.y0, rr.x1, rr.y1)))
                rows.append(cells)
            bb = fitz.Rect(t.bbox) * m
            g = Grid(rows, "pymupdf", (bb.x0, bb.y0, bb.x1, bb.y1))
            if _intersects(g.bbox, scope):
                grids.append(g)
        return grids
    finally:
        doc.close()


# ------------------------------------------------------------------ helpers

def _cell_words(lay: PageLayout, bbox: Rect) -> list[Word]:
    if bbox[2] - bbox[0] <= 0:
        return []
    ws = lay.words_in_region(bbox)
    rows = group_rows(ws)
    return [w for r in rows for w in r.words]


def _map_header(texts: list[str], columns: dict) -> tuple[dict[str, int], dict[str, float]]:
    pairs = []
    for fname, ccfg in columns.items():
        for ci, t in enumerate(texts):
            s = max(header_score(a, t) for a in ccfg["headers"])
            pairs.append((s, fname, ci))
    pairs.sort(reverse=True)
    mapping, scores, used = {}, {}, set()
    for s, fname, ci in pairs:
        if s < HEADER_MIN or fname in mapping or ci in used:
            continue
        mapping[fname], scores[fname] = ci, s
        used.add(ci)
    return mapping, scores


def _is_total(text: str, labels: list[str]) -> bool:
    t = norm(text)
    if not t:
        return False
    for lab in labels:
        n = norm(lab)
        if n and (fuzz.partial_ratio(n, t) >= 90 if len(n) >= 6 else n in t.split()):
            return True
    return False


def _cell_value(lay: PageLayout, bbox: Rect, ccfg: dict, threshold: float) -> CellValue:
    ctype = ccfg.get("type", "text")
    words = _cell_words(lay, bbox)
    scanned = lay.page.is_scanned
    if ctype == "checkbox":
        d = cb.detect_in_region(lay, bbox, threshold)
        return CellValue(d.method, d.checked, d.box_bbox or bbox,
                         cb.decision_conf(d, scanned, threshold),
                         words, {"method": d.method, "score": round(d.score, 3), "checked": d.checked})
    if ctype == "checkbox_group":
        opts = options_map(ccfg["options"])
        g = cb.detect_group(lay, list(opts), bbox, threshold)
        val = opts[g.selected[0]] if len(g.selected) == 1 else None
        conf = min((cb.decision_conf(d, scanned, threshold) for d in g.decisions), default=0.3)
        return CellValue(", ".join(g.selected) or None, val, union(words) or bbox,
                         conf if not g.ambiguous else min(conf, 0.3), words,
                         {"selected": g.selected, "ambiguous": g.ambiguous,
                          "decisions": [d.as_dict() for d in g.decisions]})
    text = join(words) if words else None
    cv = CellValue(text, text, union(words), mean_conf(words), words)
    if scanned and text:
        # OCR merges neighbouring columns into one box ("1234567890$12,750.01");
        # such a box sticks out of the cell. Re-reading the cell alone splits it.
        merged = any(w.x1 > bbox[2] + 1 or w.x0 < bbox[0] - 1 for w in words)
        shaped = ctype not in CELL_PATTERNS or bool(CELL_PATTERNS[ctype].fullmatch(text))
        glued = bool(re.search(r"[a-z][A-Z]", text))        # "FirstNationalBank"
        if merged or glued or not shaped or cv.conf < REOCR_BELOW_CONF:
            _reocr_cell(lay, bbox, ctype, cv, merged=merged, glued=glued)
    # OCR'd typed cells that fail their shape, or were read with low confidence,
    # are re-read on their own (tight crop, whitelisted characters).
    if scanned and text and ctype in CELL_PATTERNS and (
            not CELL_PATTERNS[ctype].fullmatch(text) or cv.conf < REOCR_BELOW_CONF):
        _reocr_cell(lay, bbox, ctype, cv)
    return cv


REOCR_BELOW_CONF = 0.92


# Shape a typed cell must have; OCR'd cells that fail are re-read on their own.
CELL_PATTERNS = {
    "account": re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]{3,}"),
    "currency": re.compile(r"[\$S]?\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?"),
}


def _reocr_cell(lay: PageLayout, bbox: Rect, ctype: str, cv: CellValue, merged: bool = False,
                glued: bool = False):
    """Read one cell on its own. A merged cell needs detection to split it."""
    region = bbox if merged or not cv.bbox else cv.bbox
    rect = (region[0] - 2, region[1] - 2, region[2] + 4, region[3] + 2)
    text, conf = ocr_region(lay.page.image, lay.page.to_px(rect), fast=not merged)
    pattern = CELL_PATTERNS.get(ctype)
    if pattern:
        text = text.replace(" ", "")
    if not text or (pattern and not pattern.fullmatch(text)):
        return
    shape_ok = bool(cv.text) and bool(pattern.fullmatch(cv.text)) if pattern else bool(cv.text)
    recovered_spaces = glued and text.count(" ") > (cv.text or "").count(" ")
    if merged or recovered_spaces or not shape_ok or conf > cv.conf:
        cv.text = cv.value = text
        cv.conf = conf
        cv.reocr = True


def _row_empty(row: dict[str, CellValue], key_columns: list[str], columns: dict) -> bool:
    for k in key_columns:
        cv = row.get(k)
        if cv is None:
            continue
        if columns[k].get("type") in ("checkbox", "checkbox_group"):
            continue
        if cv.text and norm(cv.text):
            return False
    return True


# ------------------------------------------------------------------ grid path

def _from_grid(lay: PageLayout, grid: Grid, tcfg: dict, threshold: float) -> TableResult | None:
    columns = tcfg["columns"]
    total_cfg = tcfg.get("total_row") or {}
    best = None
    for hi, row in enumerate(grid.rows[:3]):
        texts = [join(_cell_words(lay, c.bbox)) for c in row]
        mapping, scores = _map_header(texts, columns)
        if best is None or len(mapping) > len(best[1]):
            best = (hi, mapping, scores)
    if best is None or len(best[1]) < max(2, int(0.6 * len(columns))):
        return None
    hi, mapping, scores = best
    rows, totals = [], {}
    for row in grid.rows[hi + 1:]:
        row_text = join([w for c in row for w in _cell_words(lay, c.bbox)])
        if total_cfg and _is_total(row_text, total_cfg.get("labels", [])):
            for out_name, col in (total_cfg.get("values") or {}).items():
                if col in mapping:
                    totals[out_name] = _cell_value(lay, row[mapping[col]].bbox, columns[col], threshold)
            break
        vals = {f: _cell_value(lay, row[mapping[f]].bbox, columns[f], threshold)
                for f in columns if f in mapping}
        if _row_empty(vals, tcfg.get("key_columns") or list(columns), columns):
            continue
        rows.append(vals)
    result = TableResult(rows, totals, grid.method, scores, lay.page_no)
    return result if _plausible(result, tcfg) else None


def _plausible(res: TableResult, tcfg: dict) -> bool:
    """A grid whose rows leave a mapped key column completely empty was mis-segmented."""
    if not res.rows:
        return True
    columns = tcfg["columns"]
    for k in tcfg.get("key_columns") or list(columns):
        if columns[k].get("type") in ("checkbox", "checkbox_group") or k not in res.rows[0]:
            continue
        if not any(r[k].text and norm(r[k].text) for r in res.rows):
            return False
    return True


# ------------------------------------------------------------------ cluster path

def _from_clusters(lay: PageLayout, scope: Rect, tcfg: dict, threshold: float) -> TableResult | None:
    columns = tcfg["columns"]
    total_cfg = tcfg.get("total_row") or {}
    # 1. candidate header matches for every column; pick the row most columns agree on
    cands: dict[str, list] = {}
    for fname, ccfg in columns.items():
        found = []
        for alias in ccfg["headers"]:
            found += lay.find_labels(alias, scope, threshold=HEADER_MIN, phrase_start=False)
        if not found:
            # OCR may have merged this header into a neighbouring one
            found = [m for m in (lay.find_label_in_token(a, scope) for a in ccfg["headers"]) if m]
        cands[fname] = found
    ys = sorted({round(m.y0) for ms in cands.values() for m in ms})
    best_y, best_n = None, 0
    for y in ys:
        n = sum(any(abs(m.y0 - y) <= 6 for m in ms) for ms in cands.values())
        if n > best_n:
            best_y, best_n = y, n
    if best_y is None or best_n < max(2, int(0.6 * len(columns))):
        return None
    header = {}
    for fname, ms in cands.items():
        row_ms = [m for m in ms if abs(m.y0 - best_y) <= 6]
        if row_ms:
            header[fname] = max(row_ms, key=lambda m: m.score)
    order = sorted(header, key=lambda f: header[f].x0)
    bounds = {}
    for i, f in enumerate(order):
        left = header[f].x0 - 4 if i else scope[0]
        right = header[order[i + 1]].x0 - 4 if i + 1 < len(order) else scope[2]
        bounds[f] = (left, right)
    header_bottom = max(m.y1 for m in header.values())
    alias_text = {f: " ".join(columns[f]["headers"]) for f in header}

    body = [w for w in lay.words_in_region((scope[0], header_bottom + 0.5, scope[2], scope[3]))]
    line_h = sorted(m.y1 - m.y0 for m in header.values())[len(header) // 2]
    rows, totals = [], {}
    prev_bottom = header_bottom
    for r in group_rows(body):
        # A long vertical gap ends the table (page footer, next block); blank
        # form rows are ~16-20pt each, so allow several of them.
        if r.y0 - prev_bottom > MAX_ROW_GAP:
            break
        prev_bottom = r.y1
        cells: dict[str, list[Word]] = {f: [] for f in header}
        for w in r.words:
            for f, (lo, hi) in bounds.items():
                if lo <= w.cx < hi:
                    cells[f].append(w)
                    break
        # wrapped header lines right under the header ("or before closing")
        if r.y0 - header_bottom < 2.2 * line_h and all(
                not ws or fuzz.partial_ratio(norm(join(ws)), norm(alias_text[f])) >= 85
                for f, ws in cells.items()):
            header_bottom = r.y1
            continue
        y_lo, y_hi = r.y0 - 3, r.y1 + 3
        if total_cfg and _is_total(r.text, total_cfg.get("labels", [])):
            for out_name, col in (total_cfg.get("values") or {}).items():
                if col in bounds:
                    lo, hi = bounds[col]
                    totals[out_name] = _cell_value(lay, (lo, y_lo, hi, y_hi), columns[col], threshold)
            break
        vals = {f: _cell_value(lay, (bounds[f][0], y_lo, bounds[f][1], y_hi), columns[f], threshold)
                for f in columns if f in header}
        if _row_empty(vals, tcfg.get("key_columns") or list(columns), columns):
            continue
        rows.append(vals)
    scores = {f: m.score for f, m in header.items()}
    return TableResult(rows, totals, "cluster", scores, lay.page_no)


# ------------------------------------------------------------------ entry point

def options_map(spec) -> dict:
    """checkbox options as {printed label: output value}."""
    return dict(spec) if isinstance(spec, dict) else {o: o for o in spec}


def extract_table(lay: PageLayout, scope: Rect, tcfg: dict, threshold: float = 0.15) -> TableResult | None:
    result = None
    for g in sorted(pymupdf_grids(lay, scope), key=lambda g: g.bbox[1]):
        # the grid must start inside the scope (tables above belong to other sections);
        # skip degenerate grids such as a lone checkbox square
        if scope[1] - 3 <= g.bbox[1] <= scope[3] and len(g.rows) >= 2 and g.bbox[2] - g.bbox[0] >= 60:
            result = _from_grid(lay, g, tcfg, threshold)
            if result:
                break
    if result is None:
        result = _from_clusters(lay, scope, tcfg, threshold)
    none_label = tcfg.get("none_checkbox")
    if none_label:
        m = lay.find_label(none_label, scope, threshold=80, phrase_start=False)
        if result is None:
            result = TableResult([], {}, "none", {}, lay.page_no)
        result.none_checked = cb.detect_option(lay, m, threshold) if m else None
    return result
