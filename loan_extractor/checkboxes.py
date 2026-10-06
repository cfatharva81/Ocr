"""Checkbox detection.

For each option the decision is made by the first method that applies:
  1. widget  - an AcroForm checkbox/radio (by configured name, or the widget
               sitting just left of the option label)
  2. glyph   - a check glyph (X, ☒, ☑, ■, ✓ ...) printed/OCR'd just left of the label
  3. opencv  - a square contour left of the label on the page image; checked
               when the ink fill ratio inside it exceeds the threshold

Every decision records which method decided it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .pdf_reader import Widget, Word
from .layout import BOX_GLYPHS, LabelMatch, PageLayout, Rect, v_overlap

CHECK_GLYPHS = BOX_GLYPHS - {"☐", "□"}      # empty-box glyphs are not checks
_GLYPH_PREFIX = re.compile(r"^[\[\(|{]?[xX☒☑■✓✔][\]\)|}]?(?=[A-Za-z])")


@dataclass
class CheckboxDecision:
    option: str
    checked: bool
    method: str                      # widget | glyph | opencv | none
    score: float = 1.0               # fill ratio for opencv, 1.0 otherwise
    box_bbox: Rect | None = None     # the checkbox itself (points)
    label_bbox: Rect | None = None
    label_score: float | None = None
    widget_name: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"option": self.option, "checked": self.checked, "method": self.method,
                "score": round(self.score, 3), "box_bbox": _r(self.box_bbox),
                "widget_name": self.widget_name}


@dataclass
class GroupResult:
    selected: list[str]
    decisions: list[CheckboxDecision] = field(default_factory=list)
    missing_options: list[str] = field(default_factory=list)

    @property
    def ambiguous(self) -> bool:
        return len(self.selected) != 1

    @property
    def methods(self) -> list[str]:
        return sorted({d.method for d in self.decisions})


def _r(b):
    return None if b is None else tuple(round(v, 1) for v in b)


def is_check_glyph(text: str) -> bool:
    t = text.strip().strip("[](){}|").strip()
    return t.lower() in CHECK_GLYPHS


# ------------------------------------------------------------------ widgets

def _widget_for(label: LabelMatch, widgets: list[Widget], name: str | None) -> Widget | None:
    checkable = [w for w in widgets if w.is_checkbox]
    if name:
        for w in checkable:
            if w.name == name:
                return w
    best, best_dx = None, 1e9
    for w in checkable:
        x0, y0, x1, y1 = w.rect
        dx = label.x0 - x1
        if -2 <= dx <= 20 and v_overlap(y0, y1, label.y0, label.y1) > 0.3 and dx < best_dx:
            best, best_dx = w, dx
    return best


# ------------------------------------------------------------------ glyphs

def _glyph_left_of(lay: PageLayout, label: LabelMatch, reach: float = 22.0) -> Word | None:
    first = label.words[0]
    if _GLYPH_PREFIX.match(first.text):          # OCR glued "[X]Own"
        return first
    for w in lay.page.words:
        if w in label.words:
            continue
        if (label.x0 - reach <= w.x0 and w.x1 <= label.x0 + 1.5
                and v_overlap(w.y0, w.y1, label.y0, label.y1) >= 0.3 and is_check_glyph(w.text)):
            return w
    return None


# ------------------------------------------------------------------ opencv

def _binary(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


def find_boxes(lay: PageLayout, region: Rect, min_side: float = 5.0, max_side: float = 16.0
               ) -> list[Rect]:
    """Square contours inside `region` (points) on the page image, as point rects."""
    page = lay.page
    s = page.scale
    h_img, w_img = page.image.shape[:2]
    x0, y0, x1, y1 = (int(max(0, region[0] * s)), int(max(0, region[1] * s)),
                      int(min(w_img, region[2] * s)), int(min(h_img, region[3] * s)))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return []
    crop = page.image[y0:y1, x0:x1]
    bw = _binary(crop)
    # Close small gaps in noisy scans so box outlines form closed contours.
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        bx, by, bw_, bh = cv2.boundingRect(c)
        side_w, side_h = bw_ / s, bh / s
        if not (min_side <= side_w <= max_side and min_side <= side_h <= max_side):
            continue
        if not 0.75 <= side_w / side_h <= 1.33:
            continue
        # outline must be roughly rectangular: 4-ish vertices after simplification,
        # or (noisy scans) an outer contour that encloses most of its bounding box.
        # An "X" glyph fails both tests.
        approx = cv2.approxPolyDP(c, 0.08 * cv2.arcLength(c, True), True)
        solid = cv2.contourArea(c) >= 0.6 * bw_ * bh
        if not (4 <= len(approx) <= 8 or solid):
            continue
        boxes.append(((x0 + bx) / s, (y0 + by) / s, (x0 + bx + bw_) / s, (y0 + by + bh) / s))
    # de-duplicate inner/outer contours of the same box (keep the outer one)
    boxes.sort(key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
    out: list[Rect] = []
    for b in boxes:
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        if not any(o[0] <= cx <= o[2] and o[1] <= cy <= o[3] for o in out):
            out.append(b)
    return out


def fill_ratio(lay: PageLayout, box: Rect, inset: float = 0.2) -> float:
    """Fraction of dark pixels inside the box, excluding its border."""
    page = lay.page
    s = page.scale
    w, h = box[2] - box[0], box[3] - box[1]
    dx, dy = max(w * inset, 1.2), max(h * inset, 1.2)
    x0, y0 = int((box[0] + dx) * s), int((box[1] + dy) * s)
    x1, y1 = int((box[2] - dx) * s), int((box[3] - dy) * s)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    bw = _binary(page.image[max(0, y0 - 2):y1 + 2, max(0, x0 - 2):x1 + 2])
    inner = bw[2:-2, 2:-2] if bw.shape[0] > 4 and bw.shape[1] > 4 else bw
    return float((inner > 0).mean()) if inner.size else 0.0


def _opencv_left_of(lay: PageLayout, label: LabelMatch, threshold: float, reach: float = 24.0
                    ) -> tuple[bool, float, Rect] | None:
    boxes = getattr(lay, "_checkbox_boxes", None)
    if boxes is None:
        boxes = find_boxes(lay, lay.full_rect)
        lay._checkbox_boxes = boxes
    cy = (label.y0 + label.y1) / 2
    boxes = [b for b in boxes
             if label.x0 - reach <= b[2] and b[0] <= label.x0 + 12
             and abs((b[1] + b[3]) / 2 - cy) <= 10]
    if not boxes:
        return None
    box = min(boxes, key=lambda b: abs((b[0] + b[2]) / 2 - (label.x0 + 4)))
    ratio = fill_ratio(lay, box)
    return ratio >= threshold, ratio, box


# ------------------------------------------------------------------ public API

def decision_conf(d: CheckboxDecision, scanned: bool, threshold: float = 0.15) -> float:
    """Source confidence of one checkbox decision.

    widget 1.0; glyph 0.95 native / 0.9 OCR; OpenCV 0.75-0.95 depending on how
    far the fill ratio is from the threshold; 0.3 when no box was located.
    """
    if d.method == "widget":
        return 1.0
    if d.method == "glyph":
        return 0.9 if scanned else 0.95
    if d.method == "opencv":
        margin = min(1.0, abs(d.score - threshold) / max(threshold, 1e-6))
        return 0.75 + 0.2 * margin
    return 0.3


def detect_option(lay: PageLayout, label: LabelMatch, threshold: float = 0.15,
                  widget_name: str | None = None) -> CheckboxDecision:
    opt = label.label
    w = _widget_for(label, lay.page.widgets, widget_name)
    if w is not None:
        return CheckboxDecision(opt, w.checked, "widget", 1.0, w.rect, label.bbox, label.score, w.name)
    g = _glyph_left_of(lay, label)
    if g is not None:
        return CheckboxDecision(opt, True, "glyph", 1.0, g.bbox, label.bbox, label.score)
    cv = _opencv_left_of(lay, label, threshold)
    if cv is not None:
        checked, ratio, box = cv
        return CheckboxDecision(opt, checked, "opencv", ratio, box, label.bbox, label.score)
    return CheckboxDecision(opt, False, "none", 0.0, None, label.bbox, label.score)


def detect_group(lay: PageLayout, options: list[str], band: Rect | None,
                 threshold: float = 0.15, widget_names: dict[str, str] | None = None,
                 label_threshold: float = 80) -> GroupResult:
    """Decide every option of a group inside `band` (points).

    Longer options are matched first and their words excluded, so "Deposited"
    never matches inside "Not Deposited".
    """
    widget_names = widget_names or {}
    used: set[int] = set()
    matches: dict[str, LabelMatch] = {}
    for opt in sorted(options, key=len, reverse=True):
        # Very short options ("NO", "YES") lose a lot of ratio to a single OCR
        # slip ("ves"); the band keeps the lower threshold safe.
        thr = label_threshold if len(opt) > 3 else min(label_threshold, 60)
        found = lay.find_labels(opt, scope=band, threshold=thr, phrase_start=False, exclude=used)
        if found:
            m = min(found, key=lambda m: (-m.score, m.x0))
            matches[opt] = m
            used |= {id(w) for w in m.words}
    decisions, missing = [], []
    for opt in options:
        if opt not in matches:
            missing.append(opt)
            continue
        decisions.append(detect_option(lay, matches[opt], threshold, widget_names.get(opt)))
    selected = [d.option for d in decisions if d.checked]
    return GroupResult(selected, decisions, missing)


def detect_in_region(lay: PageLayout, region: Rect, threshold: float = 0.15) -> CheckboxDecision:
    """Single checkbox somewhere inside a region (e.g. a table cell)."""
    for w in lay.page.widgets:
        if w.is_checkbox:
            cx, cy = (w.rect[0] + w.rect[2]) / 2, (w.rect[1] + w.rect[3]) / 2
            if region[0] <= cx <= region[2] and region[1] <= cy <= region[3]:
                return CheckboxDecision("cell", w.checked, "widget", 1.0, w.rect, None, None, w.name)
    for w in lay.words_in_region(region):
        if is_check_glyph(w.text):
            return CheckboxDecision("cell", True, "glyph", 1.0, w.bbox)
    boxes = find_boxes(lay, region)
    if boxes:
        box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        ratio = fill_ratio(lay, box)
        return CheckboxDecision("cell", ratio >= threshold, "opencv", ratio, box)
    return CheckboxDecision("cell", False, "none", 0.0, None)
