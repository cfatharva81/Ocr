"""FastAPI service.

    uvicorn loan_extractor.api:app --port 8000
    curl -F "file=@data/loan_application.pdf" "http://localhost:8000/extract?with_metadata=true"
    curl -F "file=@data/loan_application.pdf" "http://localhost:8000/extract-llm"   # needs GEMINI_API_KEY
    curl -F "file=@any_document.pdf" "http://localhost:8000/extract-llm?mode=auto"  # Gemini picks the fields
    curl -F "file=@doc.pdf" -F "template=@templates/my_form.json" "http://localhost:8000/extract-llm?mode=template"
"""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse

from . import __version__
from .extractor import extract, setup_logging
from .llm_extractor import DEFAULT_MODEL, SUPPORTED_MODELS, check_model, extract_with_llm, load_template
from .pdf_reader import ocr_available

setup_logging(logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(title="URLA / Form 1003 extractor", version=__version__,
              description="PDF-to-JSON extraction: rule-based (PDF parsing, OCR, layout, regex) or OCR + Gemini + Pydantic.")


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
  <select id="mode">
    <option value="rules">Rule-based</option>
    <option value="llm">OCR + Gemini (loan schema)</option>
    <option value="auto">OCR + Gemini (auto fields, any document)</option>
    <option value="ocr">OCR text only</option>
  </select>
  <select id="model" title="Gemini model"></select>
  <label><input type="checkbox" id="meta"> with metadata</label>
  <button>Extract</button>
</form>
<p id="status"></p>
<div id="ocrbox" hidden><h3>Raw OCR text</h3><pre id="ocr"></pre></div>
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
  const ocr = document.getElementById('ocr'), ocrbox = document.getElementById('ocrbox');
  status.textContent = 'Extracting... (scanned PDFs take ~20 s)'; out.textContent = ''; ocr.textContent = '';
  const mode = document.getElementById('mode').value, meta = document.getElementById('meta').checked;
  const url = mode === 'rules' ? '/extract?with_metadata=' + meta
            : '/extract-llm?ocr_only=' + (mode === 'ocr') + '&mode=' + (mode === 'auto' ? 'auto' : 'schema')
              + '&model=' + encodeURIComponent(document.getElementById('model').value);
  const r = await fetch(url, {method: 'POST', body: new FormData(e.target)});
  const data = await r.json();
  status.textContent = r.ok ? 'Done' : 'Error ' + r.status;
  ocrbox.hidden = !('raw_ocr_text' in data);
  if ('raw_ocr_text' in data) { ocr.textContent = data.raw_ocr_text; delete data.raw_ocr_text; }
  out.textContent = JSON.stringify(data, null, 2);
};
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return UPLOAD_PAGE


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__, "ocr": "rapidocr" if ocr_available() else "unavailable"}


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


@app.get("/models")
def models() -> dict:
    """The Gemini models /extract-llm accepts."""
    return {"default": DEFAULT_MODEL, "models": list(SUPPORTED_MODELS)}


@app.post("/extract-llm")
async def extract_llm_endpoint(file: UploadFile = File(...),
                               force_ocr: bool = Query(False),
                               ocr_only: bool = Query(False, description="return the OCR text without calling Gemini"),
                               model: str | None = Query(None),
                               mode: str = Query("schema", pattern="^(schema|auto|template)$",
                                                 description="schema: loan form fields; auto: Gemini decides the "
                                                             "fields (response includes `template`); template: "
                                                             "fill the uploaded template"),
                               template: UploadFile | None = File(None, description="template JSON for mode=template"),
                               max_pages: int | None = Query(None, ge=1, le=100)) -> dict:
    """OCR -> Gemini -> Pydantic. The response starts with `raw_ocr_text`, then `data`."""
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
        kwargs = {"force_ocr": force_ocr, "ocr_only": ocr_only, "mode": mode, "max_pages": max_pages}
        if mode == "template" and not ocr_only:
            if template is None:
                raise HTTPException(400, "mode=template needs a template file")
            try:
                kwargs["template"] = load_template(json.loads(await template.read()))
            except ValueError as ex:      # bad JSON or not a valid template
                raise HTTPException(400, f"invalid template: {ex}") from ex
        if model:
            kwargs["model"] = model
        try:
            out = await run_in_threadpool(extract_with_llm, path, **kwargs)
        except Exception as ex:  # noqa: BLE001
            log.exception("LLM extraction failed")
            raise HTTPException(422, f"extraction failed: {type(ex).__name__}: {ex}") from ex
    out["extraction_metadata"]["source_file"] = file.filename
    return out
