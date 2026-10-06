"""Command line entry point.

    python cli.py data/loan_application.pdf -o output/result.json [--pages N] [--with-metadata] [--debug]
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
    ap.add_argument("pdf", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=Path("output/result.json"))
    ap.add_argument("-n", "--pages", type=int,
                    help="number of pages to process (prompted when omitted)")
    ap.add_argument("--with-metadata", action="store_true",
                    help="each field as {value, confidence, page, source_text, bbox, strategy_used}")
    ap.add_argument("--debug", action="store_true", help="save page images with field boxes to --debug-dir")
    ap.add_argument("--debug-dir", type=Path, default=Path("output/debug"))
    ap.add_argument("--config", default=None, help="form field map YAML (default: urla_1003.yaml)")
    ap.add_argument("--force-ocr", action="store_true", help="OCR every page even if it has a text layer")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    setup_logging(logging.DEBUG if a.verbose else logging.INFO)
    if not a.pdf.exists():
        print(f"error: {a.pdf} not found", file=sys.stderr)
        return 2
    if a.pages is None:
        try:
            answer = input("How many pages do you want to extract? ").strip()
        except EOFError:
            print("error: --pages is required when stdin is not interactive", file=sys.stderr)
            return 2
        try:
            a.pages = int(answer)
        except ValueError:
            print("error: pages must be a positive integer", file=sys.stderr)
            return 2
    if a.pages is not None and a.pages < 1:
        print("error: pages must be a positive integer", file=sys.stderr)
        return 2
    res = extract(a.pdf, config=a.config, force_ocr=a.force_ocr,
                  debug_dir=a.debug_dir if a.debug else None, max_pages=a.pages)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(res.to_dict(a.with_metadata), indent=2), encoding="utf-8")
    print(f"wrote {a.output} ({res.metadata['processing_time_s']}s, {len(res.missing_fields)} missing, "
          f"{len(res.low_confidence_fields)} low-confidence, schema_valid={res.metadata['schema_valid']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
