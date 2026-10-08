"""Dynamic-schema extraction: PDF -> OCR/text -> Gemini -> Pydantic validation.

Needs GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment or in a .env file at the project
root (see .env.example); GEMINI_MODEL overrides the model. Real environment variables win over .env.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from statistics import median

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from . import __version__
from .checkboxes import page_checkboxes, words_with_marks
from .dynamic_schema import DynamicDocument
from .layout import PageLayout
from .pdf_reader import Page, Word, load_pdf

log = logging.getLogger(__name__)

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Text models that answered a structured-output request with this project's key on 2026-10-06.
# models.list() also returns retired ones (gemini-2.5-* now 404) and non-text ones (TTS, image,
# music, agents), so only these are accepted; `python cli.py --list-models` prints them.
SUPPORTED_MODELS = (
    "gemini-3.8-flash",         # default: newest Flash
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",    # fastest / cheapest
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",   # strongest, slower
    "gemini-flash-latest",      # aliases Google moves to newer models
    "gemini-pro-latest",
)
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL") or SUPPORTED_MODELS[0]
_OCR_NOTES = """The input is OCR text, one "=== Page N ===" block per page. Lines keep the printed order;
wide gaps between words are shown as 4+ spaces and usually separate a label from its value or one column from the next."""

_COMMON_RULES = """- Use only values present in the text. Never guess or invent values.
- Printed labels and instructions are not values: "Street", "City", "$" or "Enter amount" alone means the field is empty.
- Square checkboxes and circular radio buttons are detected from the page image and are written as
    "[X]" (checked) or "[ ]" (unchecked)
  just before their option's text, e.g. "Marital Status [X] Married [ ] Separated [ ] Unmarried".
  Only treat an option as selected when it has "[X]"; if no option in a group has one, the answer is null.
- A single box on its own (e.g. "[ ] Check if you don't have any other assets", "[X] Does not apply")
  is a yes/no answer: "Yes" for [X], "No" for [ ]. It is never null, the mark is always there.
- OCR errors are expected (O/0, l/1, S/5); fix them only when the field type makes the intended character obvious.
- Copy names, IDs, account numbers and addresses exactly as printed (after obvious OCR fixes)."""

AUTO_PROMPT = f"""You turn any document into structured data. There is no predefined field list:
you decide the sections, fields and tables from what the document contains.
{_OCR_NOTES}

How to structure it:
- Follow the document's own layout: one section per printed section or logical group (e.g. personal info, employment, totals).
- Every label/value pair becomes a field. Choose short, descriptive snake_case keys (borrower_name, not field_1);
  keep keys unique within a section.
- Repeating rows (accounts, line items, liabilities) become a table, not numbered fields.
- For a checkbox group, make one field whose value is the selected option's text (or null if none is marked);
  for a single box on its own, value_type "boolean".
- Pick the value_type that fits the value; put the value exactly as printed (keep "$" and commas, the code converts them).
- Include blank fields with value null only when the label clearly asks for data; skip headings, instructions and footers.

Rules:
{_COMMON_RULES}"""


# ====================================================================== OCR text

def page_text(page: Page, words: list[Word] | None = None) -> str:
    """The page's words as text lines in reading order (top-to-bottom, left-to-right)."""
    words = sorted(page.words if words is None else words, key=lambda w: (w.cy, w.x0))
    if not words:
        return ""
    tol = 0.5 * median(w.h for w in words)
    lines: list[list[Word]] = []
    for w in words:
        if lines and abs(w.cy - sum(x.cy for x in lines[-1]) / len(lines[-1])) <= tol:
            lines[-1].append(w)
        else:
            lines.append([w])
    out = []
    for line in lines:
        line.sort(key=lambda w: w.x0)
        parts = [line[0].text]
        for prev, w in zip(line, line[1:]):
            parts.append("    " if w.x0 - prev.x1 > 2 * max(prev.h, 1.0) else " ")
            parts.append(w.text)
        out.append("".join(parts))
    return "\n".join(out)


def ocr_text(pages: list[Page], marks: dict[int, list] | None = None) -> str:
    """All pages as text; with `marks` (page_no -> checkbox decisions) checkboxes appear as [X] / [ ]."""
    def words(p):
        return words_with_marks(PageLayout(p), marks[p.page_no]) if marks else None
    return "\n\n".join(f"=== Page {p.page_no} ({p.page_type}) ===\n{page_text(p, words(p))}" for p in pages)


def read_pdf_text(pdf_path, force_ocr: bool = False, max_pages: int | None = None,
                  threshold: float = 0.15) -> tuple[str, list[Page], dict]:
    """OCR (or read the text layer of) the pages and return the raw text.

    Read up to max_pages pages (default 20); no document-specific field map is used.
    Checkboxes are found on each page image and written into the text as [X] / [ ].
    Returns (text, pages, page_no -> checkbox decisions).
    """
    pages = load_pdf(pdf_path, max_pages=max_pages or 20, force_ocr=force_ocr)
    marks = {p.page_no: page_checkboxes(PageLayout(p), threshold) for p in pages}
    return ocr_text(pages, marks), pages, marks


# ====================================================================== Gemini

def _gemini_schema(model: type[BaseModel]) -> dict:
    """The model's JSON schema minus keywords Gemini may reject; Pydantic still enforces them."""
    def strip(node):
        if isinstance(node, dict):
            return {k: strip(v) for k, v in node.items() if k not in ("exclusiveMinimum", "exclusiveMaximum")}
        if isinstance(node, list):
            return [strip(v) for v in node]
        return node
    return strip(model.model_json_schema())


def check_model(model: str) -> str:
    model = model.removeprefix("models/")
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Gemini model {model!r} is not supported; use one of: {', '.join(SUPPORTED_MODELS)}")
    return model


def call_gemini(text: str, schema: type[BaseModel], system_prompt: str, model: str = DEFAULT_MODEL,
                api_key: str | None = None) -> str:
    """Send the OCR text to Gemini, constrained to `schema`; returns its JSON reply as a string."""
    from google import genai
    from google.genai import types

    model = check_model(model)
    client = genai.Client(api_key=api_key) if api_key else genai.Client()
    resp = client.models.generate_content(
        model=model,
        contents=f"OCR text of the document:\n\n{text}",
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_json_schema=_gemini_schema(schema),
            temperature=0,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )
    if not resp.text:
        reason = resp.candidates[0].finish_reason if resp.candidates else resp.prompt_feedback
        raise RuntimeError(f"Gemini returned no text ({reason})")
    return resp.text


# ====================================================================== pipeline

def extract_with_llm(pdf_path, force_ocr: bool = False, model: str = DEFAULT_MODEL,
                     api_key: str | None = None, max_pages: int | None = None) -> dict:
    """Return dynamically inferred document fields and their extracted values."""
    model = check_model(model)
    t0 = time.perf_counter()
    text, pages, marks = read_pdf_text(pdf_path, force_ocr, max_pages)
    out = {"raw_ocr_text": text, "data": None, "validation_errors": []}
    meta = {
        "extractor_version": __version__,
        "method": "ocr+llm:dynamic",
        "llm_model": model,
        "source_file": Path(pdf_path).name,
        "ocr_engine": "rapidocr" if any(p.is_scanned for p in pages) else None,
        "pages": [{"page_no": p.page_no, "page_type": p.page_type,
                   "ocr_mean_confidence": p.ocr_mean_conf, "word_count": len(p.words),
                   "checkboxes": len(marks[p.page_no]),
                   "checked": sum(d.checked for d in marks[p.page_no])} for p in pages],
        "checkboxes": [{"page": pn, "checked": d.checked, "method": d.method,
                        "fill_ratio": d.score if d.method in ("opencv", "radio") else None,
                        "bbox": [round(v, 1) for v in d.box_bbox]}
                       for pn, ds in marks.items() for d in ds],
    }
    reply = call_gemini(text, DynamicDocument, AUTO_PROMPT, model, api_key)
    try:
        doc = DynamicDocument.model_validate_json(reply)
        out["data"] = doc.to_nested()
        out["fields"] = doc.model_dump(mode="json")
    except ValidationError as e:
        log.warning("LLM output failed validation: %d errors", e.error_count())
        try:
            out["data"] = json.loads(reply)
        except json.JSONDecodeError:
            out["data"] = {"_unparsed_reply": reply}
        out["validation_errors"] = [{"field": ".".join(map(str, err["loc"])), "error": err["msg"]}
                                    for err in e.errors()]
    meta["schema_valid"] = not out["validation_errors"]
    meta["processing_time_s"] = round(time.perf_counter() - t0, 3)
    out["extraction_metadata"] = meta
    return out
