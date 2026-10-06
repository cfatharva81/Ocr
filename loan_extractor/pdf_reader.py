"""Read a PDF into upright pages of words with bounding boxes.

Native pages use the PDF text layer (plus AcroForm widgets). Scanned pages are
rendered, turned upright with Tesseract OSD, deskewed with OpenCV and OCR'd
with Tesseract. Either way every word ends up in the same coordinate system:
PDF points (1/72 inch) on an upright page.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pymupdf as fitz

log = logging.getLogger(__name__)

# ZapfDingbats characters that AcroForm checkboxes use for "checked".
DINGBAT_CHECKS = {"3": "\u2713", "4": "\u2714", "5": "\u2717", "6": "\u2718", "7": "\u2717",
                  "8": "\u2718", "l": "\u25cf", "n": "\u25a0", "u": "\u25c6"}


# --------------------------------------------------------------------- data

@dataclass
class Word:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    conf: float = 1.0          # 0..1; 1.0 for native text
    source: str = "native"     # native | ocr

    @property
    def bbox(self):
        return (self.x0, self.y0, self.x1, self.y1)

    @property
    def cx(self):
        return (self.x0 + self.x1) / 2

    @property
    def cy(self):
        return (self.y0 + self.y1) / 2

    @property
    def h(self):
        return self.y1 - self.y0


@dataclass
class Widget:
    """An AcroForm field."""
    name: str
    field_type: str            # Text | CheckBox | RadioButton | ...
    value: Any
    rect: tuple
    on_state: str | None = None

    @property
    def is_checkbox(self) -> bool:
        return self.field_type in ("CheckBox", "RadioButton")

    @property
    def checked(self) -> bool:
        v = self.value
        if isinstance(v, bool):
            return v
        if v in (None, "", "Off", "off", "0", 0):
            return False
        return v == self.on_state or str(v).lower() in ("yes", "on", "true", "1", "x")


@dataclass
class Page:
    page_no: int                  # 1-based
    width: float                  # points, upright
    height: float
    words: list[Word]
    image: np.ndarray             # BGR image of the upright page at `dpi`
    dpi: int
    is_scanned: bool
    file_rotation: int = 0        # /Rotate stored in the PDF
    rotation_applied: int = 0     # clockwise rotation applied to get upright content
    deskew_angle: float = 0.0
    widgets: list[Widget] = field(default_factory=list)
    ocr_mean_conf: float | None = None
    source_path: str | None = None
    render_rotation: int = 0      # /Rotate used to render a native page upright

    @property
    def scale(self) -> float:
        """Image pixels per PDF point."""
        return self.dpi / 72.0

    @property
    def page_type(self) -> str:
        return "scanned" if self.is_scanned else "native"

    def to_px(self, bbox):
        return tuple(int(round(v * self.scale)) for v in bbox)


# --------------------------------------------------------------------- native pages

def _to_bgr(pix: fitz.Pixmap) -> np.ndarray:
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        arr = arr[:, :, :3]
    elif pix.n == 1:
        arr = np.repeat(arr, 3, axis=2)
    return arr[:, :, ::-1].copy()


def _upright_rotation(page: fitz.Page) -> int:
    """The /Rotate value under which most of the text reads left-to-right."""
    votes: dict[tuple, int] = {}
    for b in page.get_text("dict")["blocks"]:
        for ln in b.get("lines", []):
            d = (round(ln["dir"][0]), round(ln["dir"][1]))
            votes[d] = votes.get(d, 0) + sum(len(s["text"]) for s in ln["spans"])
    text_dir = max(votes, key=votes.get) if votes else (1, 0)
    for rot in (0, 90, 180, 270):
        page.set_rotation(rot)
        m = page.rotation_matrix
        if (fitz.Point(text_dir) * m - fitz.Point(0, 0) * m).x > 0.9:
            return rot
    return 0


def _native_words(page: fitz.Page) -> list[Word]:
    # checkbox widgets render as ZapfDingbats "3"/"4"...; turn those into check marks
    dingbats = [fitz.Rect(s["bbox"]) for b in page.get_text("dict")["blocks"]
                for ln in b.get("lines", []) for s in ln["spans"] if "dingbat" in s["font"].lower()]
    words = []
    for x0, y0, x1, y1, text, *_ in page.get_text("words"):
        if any(fitz.Rect(x0, y0, x1, y1).intersects(d) for d in dingbats):
            text = "".join(DINGBAT_CHECKS.get(ch, ch) for ch in text)
        words.append(Word(text, x0, y0, x1, y1))
    return words


def _widgets(page: fitz.Page) -> list[Widget]:
    out = []
    for w in page.widgets() or []:
        on = None
        if w.field_type in (fitz.PDF_WIDGET_TYPE_CHECKBOX, fitz.PDF_WIDGET_TYPE_RADIOBUTTON):
            try:
                on = w.on_state()
            except Exception:  # noqa: BLE001 - malformed appearance dictionaries
                on = "Yes"
        out.append(Widget(w.field_name or "", w.field_type_string, w.field_value, tuple(w.rect), on))
    return out


# --------------------------------------------------------------------- OCR (RapidOCR)

def ocr_engine():
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR()


def ocr_available() -> bool:
    try:
        ocr_engine()
        return True
    except Exception:  # noqa: BLE001
        return False


tesseract_available = ocr_available


def _shrink(img: np.ndarray, max_side: int) -> np.ndarray:
    f = max_side / max(img.shape[:2])
    return cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else img


def detect_orientation(img: np.ndarray) -> tuple[int, list[Word]]:
    """Detect 90-degree scans by checking OCR word-box orientation."""
    words = ocr_page(img)

    def horizontal_fraction(items: list[Word]) -> float:
        return (sum(w.x1 - w.x0 > w.y1 - w.y0 for w in items) / len(items)
                if items else 0.0)

    if horizontal_fraction(words) >= 0.5:
        return 0, words

    candidates = [(0, words)]
    for angle in (90, 180, 270):
        candidates.append((angle, ocr_page(rotate(img, angle))))

    def score(candidate: tuple[int, list[Word]]) -> tuple[float, float, int]:
        _, items = candidate
        confidence = sum(w.conf for w in items) / len(items) if items else 0.0
        return horizontal_fraction(items), confidence, len(items)

    return max(candidates, key=score)


def rotate(img: np.ndarray, degrees_cw: int) -> np.ndarray:
    codes = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    code = codes.get(degrees_cw % 360)
    return cv2.rotate(img, code) if code is not None else img


def estimate_skew(img: np.ndarray, max_angle: float = 10.0) -> float:
    """Skew in degrees, measured on the form's long horizontal rules (Hough lines)."""
    gray = _shrink(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img, 1100)
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    w = gray.shape[1]
    lines = cv2.HoughLinesP(bw, 1, np.pi / 720, threshold=int(w * 0.15),
                            minLineLength=int(w * 0.25), maxLineGap=8)
    angles = []
    for x1, y1, x2, y2 in (np.asarray(lines).reshape(-1, 4) if lines is not None else []):
        if x2 != x1:
            a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(a) <= max_angle:
                angles.append(a)
    return float(-np.median(angles)) if angles else 0.0


def deskew(img: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.05:
        return img
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


def remove_rules(img: np.ndarray) -> np.ndarray:
    """Blank out long ruled lines (form rules, table grids) before OCR.

    Table borders touching the text hurt Tesseract badly (page 2 of the sample
    goes from 0.68 to 0.94 mean confidence). Checkbox squares are shorter than
    the kernel and survive.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    k = max(40, gray.shape[1] // 40)
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k)))
    out = gray.copy()
    out[cv2.dilate(horiz | vert, np.ones((3, 3), np.uint8)) > 0] = 255
    return out


def ocr_page(img: np.ndarray) -> list[Word]:
    """Word-level RapidOCR output with coordinates in pixels."""
    engine = ocr_engine()
    gray = cv2.medianBlur(remove_rules(img), 3)
    result, _ = engine(gray)
    words = []
    for box, text, conf in result or []:
        text = (text or "").strip()
        if text:
            points = np.asarray(box, dtype=float)
            x0, y0 = points.min(axis=0)
            x1, y1 = points.max(axis=0)
            tokens = []
            for part in re.findall(r"\S+", text):
                tokens.extend(token for token in re.split(
                    r"(?<=[a-z])(?=[A-Z])|(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", part
                ) if token)
            if len(tokens) <= 1:
                words.append(Word(text, x0, y0, x1, y1, float(conf), "ocr"))
                continue

            # RapidOCR often returns a whole line as one box. Approximate each
            # token's horizontal extent so labels and adjacent values can anchor
            # independently in the layout stage.
            gap_units = 0.5
            total_units = sum(len(token) for token in tokens) + gap_units * (len(tokens) - 1)
            cursor = x0
            box_width = x1 - x0
            for index, token in enumerate(tokens):
                token_width = box_width * len(token) / total_units
                words.append(Word(token, cursor, y0, cursor + token_width, y1, float(conf), "ocr"))
                cursor += token_width + (box_width * gap_units / total_units if index < len(tokens) - 1 else 0)
    return words


# Characters a typed value may contain; used to re-read single values.
WHITELIST = {"ssn": "0123456789-", "date": "0123456789/-.", "phone": "0123456789()-.",
             "zip": "0123456789-", "int": "0123456789", "currency": "$0123456789,.",
             "account": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"}


def ocr_region(img: np.ndarray, rect_px, whitelist: str | None = None) -> tuple[str, float]:
    """Re-read one small region as a single line (psm 7), optionally whitelisted.

    A tight, upscaled crop with a restricted character set is far more
    reliable for digits than page-level OCR on low-resolution scans.
    """
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(0, rect_px[0]), max(0, rect_px[1]), min(w, rect_px[2]), min(h, rect_px[3])
    if x1 - x0 < 4 or y1 - y0 < 4:
        return "", 0.0
    crop = cv2.resize(remove_rules(img[y0:y1, x0:x1]), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    crop = cv2.copyMakeBorder(crop, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
    result, _ = ocr_engine()(crop)
    toks = [(text.strip(), float(conf)) for _, text, conf in result or []
            if text and text.strip()]
    if not toks:
        return "", 0.0
    return " ".join(t for t, _ in toks), sum(c for _, c in toks) / len(toks)


# --------------------------------------------------------------------- entry point

def load_pdf(path, max_pages: int = 3, dpi: int = 200, ocr_dpi: int = 300,
             force_ocr: bool = False, min_native_words: int = 25) -> list[Page]:
    """Load the first pages as upright Pages; pages without a text layer are OCR'd."""
    doc = fitz.open(str(path))
    pages = []
    for i in range(min(max_pages, doc.page_count)):
        page = doc[i]
        file_rot = page.rotation
        page.set_rotation(0)
        words, widgets = _native_words(page), _widgets(page)

        if not force_ocr and len(words) >= min_native_words:
            # Native: word boxes are in unrotated page space; find the rotation that
            # makes the text horizontal and map words, widgets and image into it.
            rot = _upright_rotation(page)
            page.set_rotation(rot)
            if rot:
                m = page.rotation_matrix
                for w in words:
                    w.x0, w.y0, w.x1, w.y1 = fitz.Rect(w.bbox) * m
                for wd in widgets:
                    wd.rect = tuple(fitz.Rect(wd.rect) * m)
            pages.append(Page(i + 1, page.rect.width, page.rect.height, words,
                              _to_bgr(page.get_pixmap(dpi=dpi)), dpi, False, file_rot,
                              (rot - file_rot) % 360, widgets=widgets, source_path=str(path),
                              render_rotation=rot))
            log.info("page %d: native, %d words, %d widgets", i + 1, len(words), len(widgets))
            continue

        # Scanned: render as displayed, then fix orientation and skew and OCR.
        page.set_rotation(file_rot)
        img = _to_bgr(page.get_pixmap(dpi=ocr_dpi))
        osd_rot, ocr_words = detect_orientation(img)
        img = rotate(img, osd_rot)
        skew = round(estimate_skew(img), 2)
        img = deskew(img, skew)
        if abs(skew) >= 0.05:
            ocr_words = ocr_page(img)
        s = ocr_dpi / 72.0
        for w in ocr_words:
            w.x0, w.y0, w.x1, w.y1 = w.x0 / s, w.y0 / s, w.x1 / s, w.y1 / s
        mean_conf = round(float(np.mean([w.conf for w in ocr_words])), 4) if ocr_words else 0.0
        pages.append(Page(i + 1, img.shape[1] / s, img.shape[0] / s, ocr_words, img, ocr_dpi, True,
                          file_rot, (file_rot + osd_rot) % 360, skew, ocr_mean_conf=mean_conf,
                          source_path=str(path)))
        log.info("page %d: scanned, %d OCR words, mean conf %.2f, rotated %d, deskewed %.2f",
                 i + 1, len(ocr_words), mean_conf, (file_rot + osd_rot) % 360, skew)
    doc.close()
    return pages
