"""FastAPI service.

    uvicorn loan_extractor.api:app --port 8000
    curl -F "file=@data/loan_application.pdf" "http://localhost:8000/extract?with_metadata=true"
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse

from . import __version__
from .extractor import extract, setup_logging
from .pdf_reader import tesseract_available

setup_logging(logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(title="URLA / Form 1003 extractor", version=__version__,
              description="Rule-based PDF-to-JSON extraction (PDF parsing, OCR, layout, regex). No LLMs.")


UPLOAD_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Loan Application Extractor</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:900px;margin:40px auto;padding:0 16px;color:#222}
 pre{background:#f4f4f4;padding:12px;overflow:auto;max-height:70vh;border-radius:6px}
 button{padding:6px 14px}
</style></head><body>
<h2>URLA / Form 1003 extractor</h2>
<form id="f">
  <input type="file" name="file" accept="application/pdf" required>
  <label><input type="checkbox" id="meta"> with metadata</label>
  <button>Extract</button>
</form>
<p id="status"></p><pre id="out"></pre>
<p><a href="/docs">API docs</a></p>
<script>
document.getElementById('f').onsubmit = async (e) => {
  e.preventDefault();
  const status = document.getElementById('status'), out = document.getElementById('out');
  status.textContent = 'Extracting... (scanned PDFs take ~20 s)'; out.textContent = '';
  const meta = document.getElementById('meta').checked;
  const r = await fetch('/extract?with_metadata=' + meta, {method: 'POST', body: new FormData(e.target)});
  const data = await r.json();
  status.textContent = r.ok ? 'Done' : 'Error ' + r.status;
  out.textContent = JSON.stringify(data, null, 2);
};
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return UPLOAD_PAGE


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__, "tesseract": tesseract_available()}


@app.post("/extract")
async def extract_endpoint(file: UploadFile = File(...),
                           with_metadata: bool = Query(False),
                           force_ocr: bool = Query(False)) -> dict:
    data = await file.read()
    if not data.startswith(b"%PDF"):
        raise HTTPException(415, "not a PDF")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "upload.pdf"
        path.write_bytes(data)
        try:
            res = await run_in_threadpool(extract, path, None, force_ocr)
        except Exception as ex:  # noqa: BLE001
            log.exception("extraction failed")
            raise HTTPException(422, f"extraction failed: {type(ex).__name__}") from ex
    out = res.to_dict(with_metadata)
    out["extraction_metadata"]["source_file"] = file.filename
    return out
