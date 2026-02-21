from __future__ import annotations

import io
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Tuple

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from securitisation_engine.data_sources.bmw_owner_trust import (
    fetch_latest_bmw_exhibit991_to_input_xlsx,
)
from securitisation_engine.runner import run_ipd_engine

app = FastAPI(title="BMW ABS IPD Engine", version="1.0")

# Simple in-memory cache per instance (prevents hammering SEC on repeated clicks)
_CACHE_TTL_S = 30 * 60  # 30 minutes
_cache: Dict[str, Dict[str, object]] = {}  # keyed by cik


def _require_user_agent() -> str:
    ua = (os.getenv("SEC_USER_AGENT") or "").strip()
    if not ua:
        raise HTTPException(
            status_code=500,
            detail="Missing SEC_USER_AGENT env var. Set it in Render as 'YourName your.email@domain.com'.",
        )
    return ua


def _generate_pack(cik: str) -> Tuple[bytes, bytes, Dict[str, str]]:
    """
    Returns: (ipd_pack_xlsx_bytes, engine_input_xlsx_bytes, metadata)
    """
    ua = _require_user_agent()

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        engine_input = td / "bmw_input.xlsx"
        template = td / "ipd_template.xlsx"
        ipd_out = td / "bmw_ipd_pack.xlsx"

        ten_d_url, ex99_url = fetch_latest_bmw_exhibit991_to_input_xlsx(
            cik=cik, user_agent=ua, out_xlsx=str(engine_input)
        )

        # run engine -> write ipd_out
        run_ipd_engine(
            input_xlsx=str(engine_input),
            template_xlsx=str(template),
            output_xlsx=str(ipd_out),
        )

        meta = {
            "cik": cik,
            "ten_d_url": ten_d_url,
            "exhibit_99_1_url": ex99_url,
        }
        return ipd_out.read_bytes(), engine_input.read_bytes(), meta


def _get_cached_or_run(cik: str) -> Tuple[bytes, bytes, Dict[str, str]]:
    now = time.time()
    hit = _cache.get(cik)
    if hit and (now - float(hit["ts"])) < _CACHE_TTL_S:
        return hit["ipd_bytes"], hit["input_bytes"], hit["meta"]

    ipd_bytes, input_bytes, meta = _generate_pack(cik)
    _cache[cik] = {"ts": now, "ipd_bytes": ipd_bytes, "input_bytes": input_bytes, "meta": meta}
    return ipd_bytes, input_bytes, meta


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home():
    default_cik = os.getenv("DEFAULT_CIK", "2049336")
    html = f"""
    <html>
      <head><title>BMW ABS IPD Engine</title></head>
      <body style="font-family: Arial, sans-serif; margin: 40px;">
        <h2>BMW Vehicle Owner Trust — IPD Pack Generator</h2>
        <p>This service pulls the latest BMW <b>10-D</b> and <b>Exhibit 99.1</b> from EDGAR, builds inputs, runs the waterfall, and returns an IPD XLSX pack.</p>

        <h3>Download</h3>
        <ul>
          <li><a href="/ipd/bmw/latest.xlsx?cik={default_cik}">Download latest IPD pack (XLSX)</a></li>
          <li><a href="/ipd/bmw/input.xlsx?cik={default_cik}">Download generated engine input (XLSX)</a></li>
          <li><a href="/ipd/bmw/meta?cik={default_cik}">View metadata (10-D + Exhibit links)</a></li>
        </ul>

        <p><b>Tip:</b> You can pass a different CIK via <code>?cik=...</code>.</p>
      </body>
    </html>
    """
    return html


@app.get("/ipd/bmw/meta")
def meta(cik: str = Query(default="2049336")):
    _, _, out_meta = _get_cached_or_run(cik.strip())
    return JSONResponse(out_meta)


@app.get("/ipd/bmw/latest.xlsx")
def latest_xlsx(cik: str = Query(default="2049336")):
    ipd_bytes, _, meta = _get_cached_or_run(cik.strip())

    filename = f"bmw_ipd_pack_cik_{meta['cik']}.xlsx"
    return StreamingResponse(
        io.BytesIO(ipd_bytes),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/ipd/bmw/input.xlsx")
def input_xlsx(cik: str = Query(default="2049336")):
    _, input_bytes, meta = _get_cached_or_run(cik.strip())

    filename = f"bmw_engine_input_cik_{meta['cik']}.xlsx"
    return StreamingResponse(
        io.BytesIO(input_bytes),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
