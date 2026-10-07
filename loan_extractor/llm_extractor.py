"""LLM extraction: OCR text -> Gemini -> Pydantic.

    PDF -> pages (pdf_reader: text layer, or RapidOCR for scans) -> plain text in
    reading order -> Gemini, constrained to a JSON schema -> Pydantic validation

Three modes:
    schema    the fixed LoanApplication model (URLA / Form 1003 only)
    auto      any document: Gemini names the sections, fields and tables itself
              (dynamic_schema.DynamicDocument), Pydantic checks the shape and types the values;
              the result's structure can be saved as a template
    template  a saved template: Gemini fills exactly its keys, so they are the same every run

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
from .dynamic_schema import DynamicDocument, Template, fill_template, template_model
from .extractor import all_labels, load_config
from .layout import PageLayout
from .pdf_reader import Page, Word, load_pdf
from .schemas import LoanApplication

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
MODES = ("schema", "auto", "template")

_OCR_NOTES = """The input is OCR text, one "=== Page N ===" block per page. Lines keep the printed order;
wide gaps between words are shown as 4+ spaces and usually separate a label from its value or one column from the next."""

_COMMON_RULES = """- Use only values present in the text. Never guess or invent values.
- Printed labels and instructions are not values: "Street", "City", "$" or "Enter amount" alone means the field is empty.
- Checkboxes were detected from the page image and are written as "[X]" (checked) or "[ ]" (unchecked)
  just before their option's text, e.g. "Marital Status [X] Married [ ] Separated [ ] Unmarried".
  Only treat an option as selected when it has "[X]"; if no option in a group has one, the answer is null.
- A single box on its own (e.g. "[ ] Check if you don't have any other assets", "[X] Does not apply")
  is a yes/no answer: "Yes" for [X], "No" for [ ]. It is never null, the mark is always there.
- OCR errors are expected (O/0, l/1, S/5); fix them only when the field type makes the intended character obvious.
- Copy names, IDs, account numbers and addresses exactly as printed (after obvious OCR fixes)."""

SCHEMA_PROMPT = f"""You extract data from a Uniform Residential Loan Application (URLA / Fannie Mae Form 1003).
{_OCR_NOTES}

Rules:
{_COMMON_RULES}
- If a field is blank or not found, return null (or [] for lists).
- Money: plain numbers without "$" or commas (e.g. 4750.01). Dates: YYYY-MM-DD."""

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


def template_prompt(tpl: Template) -> str:
    return f"""You fill in a fixed structure for a known document type: {tpl.document_type}.
The JSON schema lists every section, field and table to fill; each field's description gives its label
as printed on the document and its value type.
{_OCR_NOTES}

How to fill it:
- Each field's value is the one printed next to or under its label; never move a value into a different field.
- Put values exactly as printed (keep "$" and commas, the code converts them). If a field is blank or its label
  is not in the document, return null.
- Tables: one item per filled-in row, in printed order; skip empty rows.
- For a checkbox group the value is the selected option's text.

Rules:
{_COMMON_RULES}"""


def load_template(template) -> Template:
    """A Template from a Template, a dict, or a path to a template JSON file."""
    if isinstance(template, Template):
        return template
    if isinstance(template, dict):
        return Template.model_validate(template)
    return Template.model_validate_json(Path(template).read_text(encoding="utf-8"))


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


def read_pdf_text(pdf_path, force_ocr: bool = False, mode: str = "schema", config=None,
                  max_pages: int | None = None, threshold: float = 0.15) -> tuple[str, list[Page], dict]:
    """OCR (or read the text layer of) the pages and return the raw text.

    schema mode reads only the form's pages and uses its printed labels to spot upside-down
    scans; auto mode knows nothing about the document, so it reads up to max_pages (default 20).
    Checkboxes are found on each page image and written into the text as [X] / [ ].
    Returns (text, pages, page_no -> checkbox decisions).
    """
    expected = None
    if mode == "schema":
        cfg = load_config(config)
        expected = all_labels(cfg) + [a for sec in cfg["form_sections"].values() for a in sec.get("anchors", [])]
        max_pages = max_pages or int(cfg.get("max_pages", 3))
        threshold = float((cfg.get("settings") or {}).get("checkbox_fill_threshold", threshold))
    pages = load_pdf(pdf_path, max_pages=max_pages or 20, force_ocr=force_ocr, expected_text=expected)
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
                     api_key: str | None = None, config=None, ocr_only: bool = False,
                     mode: str = "schema", max_pages: int | None = None, template=None) -> dict:
    """{raw_ocr_text, data, [fields, template,] validation_errors, extraction_metadata}; raw_ocr_text comes first.

    In auto mode `data` is the nested {section: {field: value}} view, `fields` keeps each
    field's printed label, raw value and value_type, and `template` is the structure to save
    for template mode. Template mode needs `template` (a Template, dict or JSON file path).
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    tpl = None
    if mode == "template":
        if template is None:
            raise ValueError("template mode needs a template (save one with auto mode first)")
        tpl = load_template(template)
    if not ocr_only:
        model = check_model(model)      # fail before the slow OCR step
    t0 = time.perf_counter()
    text, pages, marks = read_pdf_text(pdf_path, force_ocr, mode, config, max_pages)
    out = {"raw_ocr_text": text, "data": None, "validation_errors": []}
    meta = {
        "extractor_version": __version__,
        "method": "ocr_only" if ocr_only else f"ocr+llm:{mode}",
        "llm_model": None if ocr_only else model,
        "source_file": Path(pdf_path).name,
        "ocr_engine": "rapidocr" if any(p.is_scanned for p in pages) else None,
        "pages": [{"page_no": p.page_no, "page_type": p.page_type,
                   "ocr_mean_confidence": p.ocr_mean_conf, "word_count": len(p.words),
                   "checkboxes": len(marks[p.page_no]),
                   "checked": sum(d.checked for d in marks[p.page_no])} for p in pages],
        "checkboxes": [{"page": pn, "checked": d.checked, "method": d.method,
                        "fill_ratio": d.score if d.method == "opencv" else None,   # widgets have a state, not ink
                        "bbox": [round(v, 1) for v in d.box_bbox]}
                       for pn, ds in marks.items() for d in ds],
    }
    if not ocr_only:
        schema, prompt = {"auto": (DynamicDocument, AUTO_PROMPT),
                          "schema": (LoanApplication, SCHEMA_PROMPT),
                          "template": (template_model(tpl) if tpl else None, template_prompt(tpl) if tpl else "")}[mode]
        reply = call_gemini(text, schema, prompt, model, api_key)
        try:
            doc = schema.model_validate_json(reply)
            if mode == "auto":
                out["data"] = doc.to_nested()
                out["fields"] = doc.model_dump(mode="json")
                out["template"] = doc.to_template().model_dump(mode="json")
            elif mode == "template":
                out["data"] = fill_template(tpl, doc)
            else:
                out["data"] = doc.model_dump(mode="json")
        except ValidationError as e:
            # Keep the model's answer so it can be inspected, and say what failed.
            log.warning("LLM output failed validation: %d errors", e.error_count())
            try:
                out["data"] = json.loads(reply)
            except json.JSONDecodeError:
                out["data"] = {"_unparsed_reply": reply}
            out["validation_errors"] = [{"field": ".".join(map(str, err["loc"])), "error": err["msg"]}
                                        for err in e.errors()]
    meta["schema_valid"] = None if ocr_only else not out["validation_errors"]
    meta["processing_time_s"] = round(time.perf_counter() - t0, 3)
    out["extraction_metadata"] = meta
    return out
