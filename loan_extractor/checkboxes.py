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
from dataclasses import dataclass, field, replace
from typing import Any

import cv2
import numpy as np

from .pdf_reader import Widget, Word
from .layout import BOX_GLYPHS, LabelMatch, PageLayout, Rect, v_overlap

CHECK_GLYPHS = BOX_GLYPHS - {"☐", "□"}      # empty-box glyphs are not checks
_GLYPH_PREFIX = re.compile(r"^[\[\(|{]?[xX☒☑■✓✔][\]\)|}]?(?=[A-Za-z])")
_RADIO_TOKENS = {"o", "⊙", "○", "◯"}


@dataclass
class CheckboxDecision:
    option: str
    checked: bool
    method: str                      # widget | glyph | opencv | radio | none
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


def _is_radio_token(text: str) -> bool:
    return text.strip().strip("[](){}|.,:;").casefold() in _RADIO_TOKENS


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
    cy = (label.y0 + label.y1) / 2

    def aligned(b):   # a real checkbox is centred on the label's line, not above or below it
        return abs((b[1] + b[3]) / 2 - cy) <= 4.0

    boxes = [b for b in find_boxes(lay, (label.x0 - reach, cy - 10, label.x0 + 1.5, cy + 10))
             if b[2] <= label.x0 + 2 and aligned(b)]
    box = max(boxes, key=lambda b: b[2]) if boxes else None       # nearest to the label
    if box is None:
        # OCR often swallows the checkbox into the option's text box, so the
        # square can start where the label is said to begin. Only its left
        # edge may reach past the label start, which letters of the label
        # cannot do, and the left-most candidate is the box.
        wide = [b for b in find_boxes(lay, (label.x0 - reach, cy - 10, label.x0 + 16, cy + 10))
                if b[0] <= label.x0 + 6 and aligned(b)]
        if not wide:
            return None
        box = min(wide, key=lambda b: b[0])
    ratio = fill_ratio(lay, box)
    return ratio >= threshold, ratio, box


# ------------------------------------------------------------------ public API

def decision_conf(d: CheckboxDecision, scanned: bool, threshold: float = 0.15) -> float:
    """Source confidence of one checkbox decision.

    widget 1.0; glyph 0.95 native / 0.9 OCR; image-based marks 0.75-0.95 depending
    on how far the fill ratio is from the threshold; 0.3 when no mark was located.
    """
    if d.method == "widget":
        return 1.0
    if d.method == "glyph":
        return 0.9 if scanned else 0.95
    if d.method in ("opencv", "radio"):
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


# ------------------------------------------------------------------ marks for plain text
# The LLM path sends text only, so every checkbox on the page is written into the
# text as "[X]" / "[ ]" where it sits, without knowing any labels.

def side_coverage(lay: PageLayout, box: Rect, strip: float = 0.15) -> float:
    """How completely the box has ink along all four edges: for each side, the share of
    its length with ink in a strip 15% of the box deep; the weakest side is returned.

    A drawn checkbox is ~1.0 (>= 0.81 on a blurred, noisy 120 dpi scan); the round
    letters (o, e, a ...) find_boxes() also returns have empty corners and stay <= 0.65.
    """
    s = lay.page.scale
    x0, y0 = int(box[0] * s), int(box[1] * s)
    x1, y1 = int(np.ceil(box[2] * s)), int(np.ceil(box[3] * s))
    crop = lay.page.image[max(0, y0):y1, max(0, x0):x1]
    if crop.size == 0:
        return 0.0
    ink = cv2.morphologyEx(_binary(crop), cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8)) > 0   # as find_boxes
    t = max(2, int(round(strip * min(ink.shape))))
    return float(min(ink[:t].any(0).mean(), ink[-t:].any(0).mean(),
                     ink[:, :t].any(1).mean(), ink[:, -t:].any(1).mean()))


def _radio_mark(lay: PageLayout, word: Word) -> tuple[Rect, float] | None:
    """Find a circular bubble near an OCR O-token and measure ink at its center."""
    if not _is_radio_token(word.text):
        return None
    following = [w for w in lay.page.words if w is not word and -2 <= w.x0 - word.x1 <= 36
                 and v_overlap(w.y0, w.y1, word.y0, word.y1) >= 0.15]
    if not following:
        return None

    scale = lay.page.scale
    image = lay.page.image
    token_cx, token_cy = (word.x0 + word.x1) / 2, (word.y0 + word.y1) / 2
    x0, y0 = max(0, int((token_cx - 8) * scale)), max(0, int((token_cy - 8) * scale))
    x1, y1 = min(image.shape[1], int((token_cx + 8) * scale)), min(image.shape[0], int((token_cy + 8) * scale))
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(ink, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    circles = []
    for contour in contours:
        bx, by, bw, bh = cv2.boundingRect(contour)
        width, height = bw / scale, bh / scale
        if not (4.0 <= width <= 13.0 and 4.0 <= height <= 13.0 and 0.7 <= width / height <= 1.4):
            continue
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * cv2.contourArea(contour) / (perimeter * perimeter) if perimeter else 0.0
        center_x, center_y = (x0 + bx + bw / 2) / scale, (y0 + by + bh / 2) / scale
        distance = ((center_x - token_cx) ** 2 + (center_y - token_cy) ** 2) ** 0.5
        if circularity >= 0.65 and distance <= 7.0:
            circles.append((distance, -width * height, center_x, center_y, width, height))
    if not circles:
        return None

    _, _, center_x, center_y, width, height = min(circles)
    radius = max(1.0, min(width, height) * 0.2)
    px, py = round(center_x * scale) - x0, round(center_y * scale) - y0
    radius_px = round(radius * scale)
    center = gray[max(0, py - radius_px):min(gray.shape[0], py + radius_px + 1),
                  max(0, px - radius_px):min(gray.shape[1], px + radius_px + 1)]
    if center.size == 0:
        return None
    fill = float((center < 100).mean())
    box = (center_x - width / 2, center_y - height / 2,
           center_x + width / 2, center_y + height / 2)
    return box, fill


def _overlaps(a: Rect, b: Rect) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _radio_contours(lay: PageLayout, existing: list[CheckboxDecision]) -> list[CheckboxDecision]:
    """Find aligned radio bubbles in the page image, including bubbles OCR merged into text."""
    page = lay.page
    scale = page.scale
    gray = cv2.cvtColor(page.image, cv2.COLOR_BGR2GRAY) if page.image.ndim == 3 else page.image
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, width_px, height_px = cv2.boundingRect(contour)
        width, height = width_px / scale, height_px / scale
        if not (6.0 <= width <= 9.0 and 6.0 <= height <= 9.0 and 0.8 <= width / height <= 1.25):
            continue
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * cv2.contourArea(contour) / (perimeter * perimeter) if perimeter else 0.0
        if circularity < 0.65:
            continue
        box = (x / scale, y / scale, (x + width_px) / scale, (y + height_px) / scale)
        center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        area = width * height
        candidates.append((area, center, box))

    candidates.sort(key=lambda item: -item[0])
    unique = []
    for candidate in candidates:
        if any(((candidate[1][0] - other[1][0]) ** 2 + (candidate[1][1] - other[1][1]) ** 2) ** 0.5 < 1.5
               for other in unique):
            continue
        unique.append(candidate)

    radios = []
    for _, (cx, cy), box in unique:
        vertical_group = any(abs(other[1][0] - cx) <= 2.5 and 8.0 <= abs(other[1][1] - cy) <= 35.0
                             for other in unique if other[1] != (cx, cy))
        horizontal_group = any(abs(other[1][1] - cy) <= 2.5 and 8.0 <= abs(other[1][0] - cx) <= 90.0
                               for other in unique if other[1] != (cx, cy))
        text_nearby = any(
            -6.0 <= word.x0 - box[2] <= 36.0
            and v_overlap(word.y0, word.y1, box[1], box[3]) >= 0.25
            for word in page.words
        )
        if not vertical_group and not (horizontal_group and text_nearby):
            continue
        if any(d.method != "radio" and d.box_bbox and _overlaps(box, d.box_bbox) for d in existing):
            continue
        if any(d.method == "radio" and d.box_bbox and _overlaps(box, d.box_bbox) for d in existing + radios):
            continue

        radius = max(1.0, min(box[2] - box[0], box[3] - box[1]) * 0.2)
        px, py, radius_px = round(cx * scale), round(cy * scale), round(radius * scale)
        center = gray[max(0, py - radius_px):min(gray.shape[0], py + radius_px + 1),
                      max(0, px - radius_px):min(gray.shape[1], px + radius_px + 1)]
        if center.size == 0:
            continue
        fill = float((center < 100).mean())
        radios.append(CheckboxDecision("", fill >= 0.15, "radio", round(fill, 3), box))
    return radios


# What OCR makes of a box outline or the tick in it; never real letters, except a lone X.
_BOX_JUNK = set("[](){}|口区回囗□☐☒☑■●○◯✓✔✗✘βα")   # 口/区: RapidOCR reads boxes as CJK


def page_checkboxes(lay: PageLayout, threshold: float = 0.15, min_side: float = 7.0,
                    min_coverage: float = 0.73) -> list[CheckboxDecision]:
    """Every checkbox on the page, whatever its label: AcroForm widgets first, then drawn
    squares on the page image, checked when their ink fill ratio reaches `threshold`.

    Squares smaller than `min_side` points, or without ink along all four edges
    (side_coverage), are letters, not boxes.
    """
    out = [CheckboxDecision("", w.checked, "widget", 1.0, tuple(w.rect), widget_name=w.name)
           for w in lay.page.widgets if w.is_checkbox]
    for b in find_boxes(lay, lay.full_rect):
        if min(b[2] - b[0], b[3] - b[1]) < min_side or side_coverage(lay, b) < min_coverage:
            continue
        if any(_overlaps(b, d.box_bbox) for d in out):
            continue
        ratio = fill_ratio(lay, b)
        out.append(CheckboxDecision("", ratio >= threshold, "opencv", round(ratio, 3), b))
    for word in lay.page.words:
        if any(d.box_bbox and _overlaps(word.bbox, d.box_bbox) for d in out):
            continue
        radio = _radio_mark(lay, word)
        if radio is not None:
            box, ratio = radio
            out.append(CheckboxDecision("", ratio >= threshold, "radio", round(ratio, 3), box))
    out.extend(_radio_contours(lay, out))
    return out


def words_with_marks(lay: PageLayout, decisions: list[CheckboxDecision]) -> list[Word]:
    """The page's words with each checkbox as a "[X]" / "[ ]" word in its place.

    OCR word boxes often take in the checkbox ("[I am", "αNO", "[X]Own"), so a word on
    top of a box loses its leading box characters (never letters) and is moved to just
    right of the box, behind its mark. A word that is only box characters, or a lone X
    inside a box, is the box itself and is dropped. A check glyph with no box around it
    ("X", "☒", "✓") becomes "[X]".
    """
    boxes = [(d.box_bbox, d.method) for d in decisions if d.box_bbox is not None]
    out = []
    for w in lay.page.words:
        match = next(((b, method) for b, method in boxes if w.x0 < b[2] - 1 and b[0] < w.x1
                      and abs(w.cy - (b[1] + b[3]) / 2) <= (b[3] - b[1]) / 2), None)
        if match is not None:
            box, method = match
            if method == "radio" and _is_radio_token(w.text):
                continue
            if method == "radio":
                token = w.text
                width = w.x1 - w.x0
                glyph_indices = [i for i, char in enumerate(token) if char in "Oo0⊙○◯"]
                if width > 0 and glyph_indices:
                    bubble_cx = (box[0] + box[2]) / 2
                    target = (bubble_cx - w.x0) / width * len(token) - 0.5
                    index = min(glyph_indices, key=lambda i: abs(i - target))
                    if abs(index - target) <= 1.0:
                        left_text, right_text = token[:index], token[index + 1:]
                        left_x = w.x0 + width * index / len(token)
                        right_x = w.x0 + width * (index + 1) / len(token)
                        if left_text and right_text:
                            out.append(replace(w, text=left_text, x1=left_x))
                            out.append(replace(w, text=right_text, x0=right_x))
                        elif left_text:
                            out.append(replace(w, text=left_text, x1=min(w.x1, box[0] - 0.1)))
                        elif right_text:
                            out.append(replace(w, text=right_text, x0=max(w.x0, box[2] + 0.1)))
                        continue
                out.append(replace(w, x0=max(w.x0, box[2] + 0.1)))
                continue
            m = _GLYPH_PREFIX.match(w.text)
            text = w.text[m.end():] if m else w.text.lstrip("".join(_BOX_JUNK) + " ")
            if not text or (is_check_glyph(text) and w.cx <= box[2]):
                continue
            w = replace(w, text=text, x0=max(w.x0, box[2] + 0.1))
        elif is_check_glyph(w.text):
            w = replace(w, text="[X]")
        out.append(w)
    for d in decisions:
        out.append(Word("[X]" if d.checked else "[ ]", *d.box_bbox, conf=1.0, source="checkbox"))
    return out
