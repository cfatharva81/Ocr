"""Command line entry point.

    python cli.py any_document.pdf -o output/result.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from loan_extractor.logging_utils import setup_logging


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Dynamic-schema PDF-to-JSON extraction with Gemini")
    ap.add_argument("pdf", type=Path, nargs="?")
    ap.add_argument("-o", "--output", type=Path, default=Path("output/result.json"))
    ap.add_argument("--force-ocr", action="store_true", help="OCR every page, including pages with a text layer")
    ap.add_argument("--max-pages", type=int, default=None, help="maximum pages to read (default 20)")
    ap.add_argument("--model", default=None, help="Gemini model (default: $GEMINI_MODEL or gemini-3.8-flash)")
    ap.add_argument("--list-models", action="store_true", help="print the Gemini models --model accepts")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    if a.list_models:
        from loan_extractor.llm_extractor import DEFAULT_MODEL, SUPPORTED_MODELS
        for m in SUPPORTED_MODELS:
            print(m + ("  (default)" if m == DEFAULT_MODEL else ""))
        return 0
    if a.pdf is None:
        ap.error("the pdf argument is required")
    setup_logging(logging.DEBUG if a.verbose else logging.INFO)
    if not a.pdf.exists():
        print(f"error: {a.pdf} not found", file=sys.stderr)
        return 2
    return run_dynamic(a)


def run_dynamic(a) -> int:
    from loan_extractor.llm_extractor import DEFAULT_MODEL, extract_with_llm

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    try:
        out = extract_with_llm(a.pdf, force_ocr=a.force_ocr, model=a.model or DEFAULT_MODEL,
                               max_pages=a.max_pages)
    except (ValueError, OSError) as ex:     # pydantic's ValidationError is a ValueError
        print(f"error: {ex}", file=sys.stderr)
        return 2
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["data"], indent=2))
    m = out["extraction_metadata"]
    print(f"wrote {a.output} ({m['processing_time_s']}s, model={m['llm_model']}, "
          f"schema_valid={m['schema_valid']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
