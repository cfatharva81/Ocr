"""FastAPI service.

    uvicorn loan_extractor.api:app --port 8000
    curl -F "file=@data/loan_application.pdf" "http://localhost:8000/extract?with_metadata=true"
"""
from __future__ import annotations

import logging
import json
import re
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse

from . import __version__
from .extractor import extract, setup_logging
from .pdf_reader import ocr_available

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
    <label>Pages to extract: <input type="number" id="pages" min="1" value="1" required></label>
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
    const pages = document.getElementById('pages').value;
    const r = await fetch('/extract?with_metadata=' + meta + '&pages=' + pages,
        {method: 'POST', body: new FormData(e.target)});
  const data = await r.json();
    status.textContent = r.ok
        ? 'Done. PDF: ' + data.extraction_metadata.stored_pdf + ' | JSON: ' + data.extraction_metadata.stored_json
        : 'Error ' + r.status;
  out.textContent = JSON.stringify(data, null, 2);
};
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return UPLOAD_PAGE


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__, "ocr": ocr_available(), "ocr_engine": "rapidocr"}


@app.post("/extract")
async def extract_endpoint(file: UploadFile = File(...),
                           with_metadata: bool = Query(False),
                           force_ocr: bool = Query(False),
                           pages: int = Query(3, ge=1)) -> dict:
    data = await file.read()
    if not data.startswith(b"%PDF"):
        raise HTTPException(415, "not a PDF")
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    output_dir = project_root / "output"
    original_name = Path(file.filename or "upload.pdf").name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(original_name).stem).strip("._-") or "upload"
    upload_id = uuid.uuid4().hex[:8]
    stored_pdf = data_dir / f"{stem}-{upload_id}.pdf"
    stored_json = output_dir / f"{stem}-{upload_id}.json"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    stored_pdf.write_bytes(data)
    try:
        res = await run_in_threadpool(extract, stored_pdf, None, force_ocr, None, pages)
    except Exception as ex:  # noqa: BLE001
        log.exception("extraction failed")
        if type(ex).__name__ in ("ModuleNotFoundError", "ImportError"):
            raise HTTPException(503, "RapidOCR is not installed in the selected Python environment") from ex
        raise HTTPException(422, f"extraction failed: {type(ex).__name__}") from ex
    out = res.to_dict(with_metadata)
    out["extraction_metadata"]["source_file"] = file.filename
    out["extraction_metadata"]["stored_pdf"] = stored_pdf.relative_to(project_root).as_posix()
    out["extraction_metadata"]["stored_json"] = stored_json.relative_to(project_root).as_posix()
    stored_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
