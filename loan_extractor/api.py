"""FastAPI service.

    uvicorn loan_extractor.api:app --port 8000
    curl -F "file=@any_document.pdf" "http://localhost:8000/extract-llm"  # dynamic schema; needs GEMINI_API_KEY
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse

from . import __version__
from .logging_utils import setup_logging
from .llm_extractor import DEFAULT_MODEL, SUPPORTED_MODELS, check_model, extract_with_llm
from .pdf_reader import ocr_available

setup_logging(logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(title="Dynamic Document Extractor", version=__version__,
              description="Dynamic-schema PDF extraction with OCR and Gemini.")


UPLOAD_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Dynamic Document Extractor</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:900px;margin:40px auto;padding:0 16px;color:#222}
 pre{background:#f4f4f4;padding:12px;overflow:auto;max-height:70vh;border-radius:6px}
 button{padding:6px 14px}
</style></head><body>
<h2>Dynamic Document Extractor</h2>
<form id="f">
  <input type="file" name="file" accept="application/pdf" required>
  <select id="model" title="Gemini model"></select>
    <label for="max-pages">Pages to process</label>
    <input id="max-pages" type="number" min="1" max="100" value="20" required>
  <button>Extract</button>
</form>
<p id="status"></p>
<h3>OCR Text</h3><pre id="ocr"></pre>
<h3>JSON</h3><pre id="out"></pre>
<p><a href="/docs">API docs</a></p>
<script>
fetch('/models').then(r => r.json()).then(d => {
  const sel = document.getElementById('model');
  for (const m of d.models) sel.add(new Option(m + (m === d.default ? ' (default)' : ''), m, false, m === d.default));
});
document.getElementById('f').onsubmit = async (e) => {
  e.preventDefault();
  const status = document.getElementById('status'), out = document.getElementById('out');
  const ocr = document.getElementById('ocr');
  status.textContent = 'Extracting with Gemini...'; out.textContent = ''; ocr.textContent = '';
    const params = new URLSearchParams({
        model: document.getElementById('model').value,
        max_pages: document.getElementById('max-pages').value
    });
    const url = '/extract-llm?' + params;
  const r = await fetch(url, {method: 'POST', body: new FormData(e.target)});
  const data = await r.json();
  status.textContent = r.ok ? 'Done' : 'Error ' + r.status;
  ocr.textContent = data.raw_ocr_text || '';
  out.textContent = JSON.stringify(data, null, 2);
};
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return UPLOAD_PAGE


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__, "ocr": "rapidocr" if ocr_available() else "unavailable"}


@app.get("/models")
def models() -> dict:
    """The Gemini models /extract-llm accepts."""
    return {"default": DEFAULT_MODEL, "models": list(SUPPORTED_MODELS)}


@app.post("/extract-llm")
async def extract_llm_endpoint(file: UploadFile = File(...),
                               force_ocr: bool = Query(False),
                               model: str | None = Query(None),
                               max_pages: int | None = Query(None, ge=1, le=100)) -> dict:
    """OCR -> Gemini dynamic schema. The response includes the extracted data and inferred fields."""
    if model:
        try:
            model = check_model(model)
        except ValueError as ex:
            raise HTTPException(400, str(ex)) from ex
    data = await file.read()
    if not data.startswith(b"%PDF"):
        raise HTTPException(415, "not a PDF")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "upload.pdf"
        path.write_bytes(data)
        kwargs = {"force_ocr": force_ocr, "max_pages": max_pages}
        if model:
            kwargs["model"] = model
        try:
            out = await run_in_threadpool(extract_with_llm, path, **kwargs)
        except Exception as ex:  # noqa: BLE001
            log.exception("LLM extraction failed")
            raise HTTPException(422, f"extraction failed: {type(ex).__name__}: {ex}") from ex
    out["extraction_metadata"]["source_file"] = file.filename
    return out
