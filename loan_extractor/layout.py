"""Word-level layout helpers and section detection.

Words are grouped into rows (by vertical overlap) and rows into phrases
(runs of words separated by less than a gap threshold). Printed labels are
located by fuzzy matching (rapidfuzz) against word windows inside a phrase,
and values are read from a region anchored on the label: to its right, or
below it.

All coordinates are PDF points in the upright page space produced by pdf_reader.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from rapidfuzz import fuzz

from .pdf_reader import Page, Word

Rect = tuple[float, float, float, float]

# Check marks / box glyphs as they appear in text layers and OCR output.
BOX_GLYPHS = {"x", "☒", "☑", "■", "✓", "✔", "✗", "✘", "●",
              "◆", "✕", "✖", "☐", "□"}


def is_box_glyph(text: str) -> bool:
    """'X', '[X]', '[x]', '☒' ... (the bracket/pipe variants are OCR'd box borders)."""
    t = text.strip().strip("[](){}|").strip()
    return t.lower() in BOX_GLYPHS


def norm(s: str) -> str:
    """Lowercase alphanumerics separated by single spaces (for fuzzy compare)."""
    s = s.replace("–", "-").replace("—", "-").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return s.strip()


def union(words: Iterable[Word]) -> Rect | None:
    ws = list(words)
    if not ws:
        return None
    return (min(w.x0 for w in ws), min(w.y0 for w in ws),
            max(w.x1 for w in ws), max(w.y1 for w in ws))


def join(words: Sequence[Word]) -> str:
    return " ".join(w.text for w in words)


def mean_conf(words: Sequence[Word]) -> float:
    return sum(w.conf for w in words) / len(words) if words else 0.0


def contains(rect: Rect | None, w: Word) -> bool:
    if rect is None:
        return True
    return rect[0] <= w.cx <= rect[2] and rect[1] <= w.cy <= rect[3]


def v_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    """Vertical overlap as a fraction of the shorter extent."""
    inter = min(a1, b1) - max(a0, b0)
    return max(0.0, inter) / max(1e-6, min(a1 - a0, b1 - b0))


# ---------------------------------------------------------------- structures

@dataclass
class Row:
    words: list[Word]

    @property
    def y0(self) -> float:
        return min(w.y0 for w in self.words)

    @property
    def y1(self) -> float:
        return max(w.y1 for w in self.words)

    @property
    def text(self) -> str:
        return join(self.words)


@dataclass
class LabelMatch:
    words: list[Word]
    score: float                       # 0..100
    label: str
    phrase: list[Word] = field(default_factory=list)

    @property
    def bbox(self) -> Rect:
        return union(self.words)  # type: ignore[return-value]

    @property
    def text(self) -> str:
        return join(self.words)

    @property
    def x0(self): return self.bbox[0]

    @property
    def y0(self): return self.bbox[1]

    @property
    def x1(self): return self.bbox[2]

    @property
    def y1(self): return self.bbox[3]


@dataclass
class ValueMatch:
    words: list[Word]

    @property
    def text(self) -> str:
        return join(self.words)

    @property
    def bbox(self) -> Rect | None:
        return union(self.words)

    @property
    def conf(self) -> float:
        return mean_conf(self.words)


def group_rows(words: Sequence[Word], min_overlap: float = 0.5) -> list[Row]:
    """Cluster words into rows by vertical overlap (robust to mixed font sizes).

    A word is compared with the median box of each open row rather than the
    row's union box, so a tall word straddling two text lines (a checkbox, a
    bigger value) cannot chain those lines into one row.
    """
    rows: list[Row] = []
    medians: list[tuple[float, float]] = []
    for w in sorted(words, key=lambda w: (w.cy, w.x0)):
        best, best_ov = None, 0.0
        for i in range(max(0, len(rows) - 4), len(rows)):
            m0, m1 = medians[i]
            ov = v_overlap(w.y0, w.y1, m0, m1)
            if ov > best_ov:
                best, best_ov = i, ov
        if best is not None and best_ov >= min_overlap:
            r = rows[best]
            r.words.append(w)
            ys0 = sorted(x.y0 for x in r.words)
            ys1 = sorted(x.y1 for x in r.words)
            medians[best] = (ys0[len(ys0) // 2], ys1[len(ys1) // 2])
        else:
            rows.append(Row([w]))
            medians.append((w.y0, w.y1))
    for r in rows:
        r.words.sort(key=lambda w: w.x0)
    rows.sort(key=lambda r: r.y0)
    return rows


def em(w: Word) -> float:
    """Approximate font size of a word.

    Native boxes span ascender to descender (~1.37 em); Tesseract/Docling
    boxes hug the glyphs (~0.75 em for capitals, less for x-height words).
    """
    return w.h / (1.37 if w.source == "native" else 0.75)


def phrase_gap(words: Sequence[Word]) -> float:
    """Horizontal gap that separates two phrases (label | value) on a row."""
    ems = sorted(em(w) for w in words) or [8.0]
    ref = ems[min(len(ems) - 1, int(len(ems) * 0.75))]   # p75: ignore x-height-only words
    return max(3.0, 0.75 * ref)


def split_phrases(row: Row) -> list[list[Word]]:
    gap = phrase_gap(row.words)
    phrases: list[list[Word]] = []
    for w in row.words:
        if phrases and w.x0 - phrases[-1][-1].x1 <= gap:
            phrases[-1].append(w)
        else:
            phrases.append([w])
    return phrases


# ---------------------------------------------------------------- page layout

class PageLayout:
    """Row/phrase index over one Page with label and value lookups."""

    def __init__(self, page: Page, stop_labels: Iterable[str] = ()):
        self.page = page
        self.rows = group_rows(page.words)
        self.phrases: list[tuple[int, list[Word]]] = []   # (row index, words)
        for ri, r in enumerate(self.rows):
            for p in split_phrases(r):
                self.phrases.append((ri, p))
        self.stop_labels = sorted({norm(s) for s in stop_labels if norm(s)}, key=len, reverse=True)

    @property
    def page_no(self) -> int:
        return self.page.page_no

    @property
    def full_rect(self) -> Rect:
        return (0.0, 0.0, self.page.width, self.page.height)

    # ------------------------------------------------------------ labels

    def find_labels(self, label: str, scope: Rect | None = None, threshold: float = 80,
                    phrase_start: bool = True, exclude: set[int] | None = None,
                    band: Rect | None = None) -> list[LabelMatch]:
        """All non-overlapping matches of `label` scoring >= threshold, best first.

        phrase_start: the match must begin a phrase. Printed labels start a
        phrase; the same word inside a filled-in value (e.g. "Street" in
        "123 Main Street") normally does not.
        """
        target = norm(label)
        if not target:
            return []
        k = max(1, len(label.split()))       # printed words, not normalized tokens
        exclude = exclude or set()
        cands: list[LabelMatch] = []
        for _, phrase in self.phrases:
            words = [w for w in phrase if contains(scope, w) and contains(band, w)]
            if not words:
                continue
            starts = [0] if phrase_start else range(len(words))
            if phrase_start and words[0] is not phrase[0]:
                continue
            for s in starts:
                best = None
                for n in range(1, k + 2):
                    win = words[s:s + n]
                    if len(win) < n or any(id(w) in exclude for w in win):
                        continue
                    if n > 1 and (not norm(win[0].text) or is_box_glyph(win[0].text)):
                        continue    # never start on a symbol or checkbox ("@", "[]", "[X]")
                    text = norm(join(win))
                    if not text:
                        continue
                    score = fuzz.ratio(target, text)
                    # Label printed as a prefix of a longer phrase token
                    # (e.g. OCR glued "Phone(555)") gets partial credit.
                    if (score < threshold and len(target) >= 8 and len(text) > len(target)
                            and text.startswith(target)):
                        score = max(score, 90.0)
                    # ties go to the longer window so trailing symbols such as
                    # the "$" in "Property Value $" stay part of the label
                    if best is None or score >= best.score:
                        best = LabelMatch(list(win), score, label, phrase)
                if best is not None and best.score >= threshold:
                    cands.append(best)
        cands.sort(key=lambda m: (-m.score, m.y0, m.x0))
        out: list[LabelMatch] = []
        used: set[int] = set()
        for m in cands:
            ids = {id(w) for w in m.words}
            if ids & used:
                continue
            used |= ids
            out.append(m)
        return out

    def find_label(self, label: str | Sequence[str], scope: Rect | None = None,
                   threshold: float = 80, phrase_start: bool = True,
                   pick: str = "best", band: Rect | None = None) -> LabelMatch | None:
        """Best match of a label (or any of several aliases).

        pick="best" -> highest score, ties broken by reading order;
        pick="first" -> the top-most match above threshold.
        """
        labels = [label] if isinstance(label, str) else list(label)
        found: list[LabelMatch] = []
        for lab in labels:
            found.extend(self.find_labels(lab, scope, threshold, phrase_start, band=band))
        if not found:
            return None
        if pick == "first":
            return min(found, key=lambda m: (round(m.y0), m.x0))
        return max(found, key=lambda m: (m.score, -m.y0, -m.x0))

    # ------------------------------------------------------------ stop labels

    def _is_stop(self, words: Sequence[Word], i: int, own: str = "") -> bool:
        """Does a known label start at words[i]?"""
        for sl in self.stop_labels:
            if sl == own:
                continue
            k = len(sl.split())
            text = norm(join(words[i:i + k]))
            if text and fuzz.ratio(sl, text) >= 90:
                return True
        return False

    def _cut_at_stop(self, words: list[Word], own: str = "") -> list[Word]:
        """Cut the run where another printed label begins.

        Only positions after a phrase-sized gap count, so "Street" inside the
        value "123 Main Street" is kept while "5   Years" is cut before "Years".
        """
        gap = phrase_gap(words)
        for i in range(len(words)):
            if i and words[i].x0 - words[i - 1].x1 <= gap:
                continue
            if self._is_stop(words, i, own):
                return words[:i]
        return words

    # ------------------------------------------------------------ values

    def value_right_of(self, lm: LabelMatch, max_width: float = 200,
                       scope: Rect | None = None, max_gap: float | None = None) -> ValueMatch | None:
        """First run of words to the right of the label on the same row."""
        x_start = lm.x1 - 6.0
        own_ids = {id(w) for w in lm.words}
        cand = [w for w in self.page.words
                if id(w) not in own_ids and w.x0 >= x_start and w.x0 <= lm.x1 + max_width
                and v_overlap(w.y0, w.y1, lm.y0, lm.y1) >= 0.45 and contains(scope, w)]
        if not cand:
            return None
        cand.sort(key=lambda w: w.x0)
        gap = max_gap if max_gap is not None else max(phrase_gap(cand + lm.words) * 2.2, 8.0)
        runs: list[list[Word]] = [[cand[0]]]
        for w in cand[1:]:
            if w.x0 - runs[-1][-1].x1 > gap:
                runs.append([w])
            else:
                runs[-1].append(w)
        for run in runs:
            if not norm(join(run)):          # a lone "$", ":" or "#" printed after the label
                continue
            run = self._cut_at_stop(run, norm(lm.label))
            return ValueMatch(run) if run else None
        return None

    def value_below(self, lm: LabelMatch, width: float = 250, height: float = 25,
                    scope: Rect | None = None) -> ValueMatch | None:
        """Words in the first text row under the label, within [x0, x0+width]."""
        x0, x1 = lm.x0 - 3, lm.x0 + width
        own_ids = {id(w) for w in lm.words}
        cand = [w for w in self.page.words
                if id(w) not in own_ids and w.y0 >= lm.y1 - 1.5 and w.cy <= lm.y1 + height
                and w.x0 >= x0 and w.x0 <= x1 and contains(scope, w)]
        if not cand:
            return None
        rows = group_rows(cand)
        first = rows[0].words
        first = self._cut_at_stop(first, norm(lm.label))
        return ValueMatch(first) if first else None

    # ------------------------------------------------------------ regions

    def words_in_region(self, rect: Rect) -> list[Word]:
        return [w for w in self.page.words if contains(rect, w)]

    def region_text(self, rect: Rect | None) -> tuple[str, list[tuple[int, int, Word]]]:
        """Text of the region, rows joined by newlines, with char spans per word."""
        parts: list[str] = []
        spans: list[tuple[int, int, Word]] = []
        pos = 0
        for r in self.rows:
            ws = [w for w in r.words if contains(rect, w)]
            if not ws:
                continue
            if parts:
                parts.append("\n")
                pos += 1
            for i, w in enumerate(ws):
                if i:
                    parts.append(" ")
                    pos += 1
                spans.append((pos, pos + len(w.text), w))
                parts.append(w.text)
                pos += len(w.text)
        return "".join(parts), spans

    def scope_between(self, base: Rect | None, after: str | Sequence[str] | None = None,
                      before: str | Sequence[str] | None = None,
                      threshold: float = 80) -> tuple[Rect, list[str]]:
        """Narrow `base` vertically: from the `after` anchor down to the `before` anchor.

        Returns (rect, warnings). A missing anchor leaves that edge unchanged.
        """
        x0, y0, x1, y1 = base or self.full_rect
        warnings = []
        if after:
            m = self.find_label(after, (x0, y0, x1, y1), threshold, phrase_start=False, pick="best")
            if m:
                y0 = m.y0 - 1
            else:
                warnings.append(f"scope anchor not found: {after!r}")
        if before:
            m = self.find_label(before, (x0, y0 + 1, x1, y1), threshold, phrase_start=False, pick="best")
            if m:
                y1 = m.y0 - 0.5
            else:
                warnings.append(f"scope anchor not found: {before!r}")
        return (x0, y0, x1, y1), warnings


# ---------------------------------------------------------------- section detection

@dataclass
class SectionRegion:
    """Where a form section (e.g. "Section 2") sits: its page and vertical band."""
    name: str
    page_no: int
    rect: Rect
    score: float
    anchors_found: list[str] = field(default_factory=list)
    method: str = "anchor"              # anchor | page_hint


def find_sections(layouts: list[PageLayout], form_sections: dict, threshold: float = 80
                  ) -> tuple[dict[str, SectionRegion], list[str]]:
    """Locate each form section by fuzzy-matching its anchor phrases on every page.

    Sections are found by content, not page number: a section goes to the page
    where its anchors match best and runs from its top-most anchor down to the
    next section's anchor on that page. `page_hint` is only a fallback.
    """
    warnings: list[str] = []
    regions: dict[str, SectionRegion] = {}
    for name, spec in form_sections.items():
        best = None                      # (score, layout, top y, anchors found)
        for lay in layouts:
            found, top, score = [], None, 0.0
            for anchor in spec.get("anchors", []):
                m = lay.find_label(anchor, threshold=threshold, phrase_start=False)
                if m:
                    found.append(anchor)
                    score = max(score, m.score)
                    top = m.y0 if top is None else min(top, m.y0)
            score += 2.0 * len(found)    # more distinct anchors = stronger evidence
            if found and (best is None or score > best[0]):
                best = (score, lay, top, found)
        if best:
            score, lay, top, found = best
            late_partial_match = (spec.get("page_hint") == lay.page_no
                                  and top > lay.page.height * 0.4
                                  and len(found) < len(spec.get("anchors", [])))
            if late_partial_match:
                regions[name] = SectionRegion(name, lay.page_no, lay.full_rect, 0.0, found, "page_hint")
                warnings.append(f"section {name}: partial anchors start too low; used page_hint={lay.page_no}")
            else:
                regions[name] = SectionRegion(name, lay.page_no, (0.0, max(0.0, top - 2), lay.page.width,
                                                                  lay.page.height), min(100.0, score), found)
        elif spec.get("page_hint") and spec["page_hint"] <= len(layouts):
            lay = layouts[spec["page_hint"] - 1]
            regions[name] = SectionRegion(name, lay.page_no, lay.full_rect, 0.0, [], "page_hint")
            warnings.append(f"section {name}: no anchor matched, fell back to page_hint={spec['page_hint']}")
        else:
            warnings.append(f"section {name}: not found")

    # a section ends where the next section on the same page starts
    for r in regions.values():
        below = [o.rect[1] for o in regions.values()
                 if o is not r and o.page_no == r.page_no and o.rect[1] > r.rect[1] + 1]
        if below:
            r.rect = (r.rect[0], r.rect[1], r.rect[2], min(below))
    return regions, warnings
