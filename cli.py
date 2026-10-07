"""Command line entry point.

    python cli.py data/loan_application.pdf -o output/result.json [--with-metadata] [--debug]
    python cli.py data/loan_application.pdf --llm            # OCR -> Gemini -> Pydantic (loan schema)
    python cli.py any_document.pdf --llm --auto               # OCR -> Gemini picks the fields -> Pydantic
    python cli.py doc.pdf --llm --auto --save-template templates/my_form.json   # freeze the structure
    python cli.py doc.pdf --llm --template templates/my_form.json               # same keys every run
    python cli.py data/loan_application.pdf --ocr-only       # just print the raw OCR text
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from loan_extractor.extractor import extract, setup_logging


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Rule-based URLA / Form 1003 PDF-to-JSON extractor")
    ap.add_argument("pdf", type=Path, nargs="?")
    ap.add_argument("-o", "--output", type=Path, default=Path("output/result.json"))
    ap.add_argument("--with-metadata", action="store_true",
                    help="each field as {value, confidence, page, source_text, bbox, strategy_used}")
    ap.add_argument("--debug", action="store_true", help="save page images with field boxes to --debug-dir")
    ap.add_argument("--debug-dir", type=Path, default=Path("output/debug"))
    ap.add_argument("--config", default=None, help="form field map YAML (default: urla_1003.yaml)")
    ap.add_argument("--force-ocr", action="store_true", help="OCR every page even if it has a text layer")
    ap.add_argument("--llm", action="store_true",
                    help="extract with Gemini from the OCR text instead of the rules (needs GEMINI_API_KEY)")
    ap.add_argument("--auto", action="store_true",
                    help="with --llm: no predefined schema, Gemini decides the sections and fields (any document)")
    ap.add_argument("--save-template", type=Path, default=None,
                    help="with --auto: save the structure Gemini chose (keys, labels, types) as a template JSON")
    ap.add_argument("--template", type=Path, default=None,
                    help="fill a saved template: same sections and keys on every run (implies --llm)")
    ap.add_argument("--max-pages", type=int, default=None, help="pages to read in --auto mode (default 20)")
    ap.add_argument("--ocr-only", action="store_true", help="print and save the raw OCR text, no extraction")
    ap.add_argument("--model", default=None, help="Gemini model for --llm (default: $GEMINI_MODEL or gemini-3.8-flash)")
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
    if a.llm or a.ocr_only or a.template:
        return run_llm(a)
    res = extract(a.pdf, config=a.config, force_ocr=a.force_ocr, debug_dir=a.debug_dir if a.debug else None)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(res.to_dict(a.with_metadata), indent=2), encoding="utf-8")
    print(f"wrote {a.output} ({res.metadata['processing_time_s']}s, {len(res.missing_fields)} missing, "
          f"{len(res.low_confidence_fields)} low-confidence, schema_valid={res.metadata['schema_valid']})")
    return 0


def run_llm(a) -> int:
    from loan_extractor.llm_extractor import DEFAULT_MODEL, extract_with_llm

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    try:
        out = extract_with_llm(a.pdf, force_ocr=a.force_ocr, model=a.model or DEFAULT_MODEL,
                               config=a.config, ocr_only=a.ocr_only, max_pages=a.max_pages,
                               mode="template" if a.template else "auto" if a.auto else "schema",
                               template=a.template)
    except (ValueError, OSError) as ex:     # pydantic's ValidationError is a ValueError
        print(f"error: {ex}", file=sys.stderr)
        return 2
    print("=" * 30 + " RAW OCR TEXT " + "=" * 30)
    print(out["raw_ocr_text"])
    print("=" * 74)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    txt = a.output.with_suffix(".ocr.txt")
    txt.write_text(out["raw_ocr_text"], encoding="utf-8")
    if a.ocr_only:
        print(f"wrote {txt}")
        return 0
    a.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["data"], indent=2))
    if a.save_template:
        if "template" not in out:
            print("error: --save-template needs --auto and a valid result", file=sys.stderr)
            return 2
        a.save_template.parent.mkdir(parents=True, exist_ok=True)
        a.save_template.write_text(json.dumps(out["template"], indent=2), encoding="utf-8")
        print(f"saved template {a.save_template}")
    m = out["extraction_metadata"]
    print(f"wrote {txt} and {a.output} ({m['processing_time_s']}s, model={m['llm_model']}, "
          f"schema_valid={m['schema_valid']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
