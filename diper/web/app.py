"""FastAPI web app for DiPer: serves the front end in static/ and the JSON API under /api.

Run with `python run_web.py` (http://127.0.0.1:8156, local only). `scripts/build_static.py` exports every path in
STATIC_ENDPOINTS to JSON for the GitHub Pages site, so keep that list in step with the GET routes below. The upload
endpoint (POST /api/analyze) runs the repo's Python analysis (diper/web/analysis.py); on the static site static_api.js
answers it with the JS port (diper_engine.js) instead.
"""
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import diper
from diper.web.analysis import DEFAULTS, MAX_SEGMENTS_RETURNED, analyse_files

STATIC = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

app = FastAPI(title="DiPer")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

ANALYSES = [
    {"id": "origin", "name": "Plot_At_Origin", "what": "All tracks translated to start at (0, 0)."},
    {"id": "speed", "name": "Speed", "what": "Mean instantaneous speed per cell; grand average ± SEM."},
    {"id": "dir_ratio", "name": "Dir_Ratio", "what": "Directionality ratio d/D over elapsed time, and at the last point."},
    {"id": "msd", "name": "MSD", "what": "Mean square displacement (overlapping intervals, up to half the track) and α."},
    {"id": "autocorr", "name": "Autocorrel", "what": "Direction autocorrelation: mean cosine of the angle between steps."},
    {"id": "autocorr3d", "name": "Autocorrel_3D", "what": "The same in 3D, when a z column (column 7) is present."},
    {"id": "vel_cor", "name": "Vel_Cor", "what": "Normalized velocity autocorrelation ⟨v(t)·v(t+τ)⟩ / ⟨v²⟩."},
    {"id": "sparse", "name": "Sparse_Data", "what": "Keep 1 of N frames before analysing (time interval × N)."},
]


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info():
    """Project summary, analyses and the defaults the repo's run_* functions use."""
    return {
        "title": "DiPer",
        "description": "A Python implementation of the DiPer toolset for directional persistence of cell migration "
                       "(originally Excel VBA macros by Gorelik & Gautreau, Nature Protocols 2014).",
        "repo": "diper_clone",
        "version": diper.__version__,
        "defaults": DEFAULTS,
        "max_segments": MAX_SEGMENTS_RETURNED,
        "analyses": ANALYSES,
        "reference": {"text": "Gorelik, R. & Gautreau, A. (2014). Quantitative and unbiased analysis of directional "
                              "persistence in cell migration. Nature Protocols 9(8), 1931-1943.",
                      "url": "https://doi.org/10.1038/nprot.2014.131"},
    }


@app.post("/api/analyze")
async def analyze(files: List[UploadFile] = File(...), time_interval: float = Form(...),
                  max_intervals: int = Form(30), sparse_n: int = Form(1), um_per_px: Optional[str] = Form(None),
                  weighting: str = Form("displacements")):
    """Analyse uploaded CSV/Excel files with the repo's Python code (see diper/web/analysis.py)."""
    payload, total = [], 0
    for f in files:
        data = await f.read()
        total += len(data)
        payload.append((f.filename or "upload.csv", data))
    if total > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Files too large for the web app (10 MB in total); use the Python library for big data.")
    try:
        px = float(um_per_px) if um_per_px not in (None, "") else None
        return analyse_files(payload, time_interval, max_intervals, sparse_n, px, weighting)
    except ValueError as e:
        raise HTTPException(400, str(e))


# GET paths (without the /api prefix) written to the static site as api/<path>.json.
STATIC_ENDPOINTS = ["/info"]
