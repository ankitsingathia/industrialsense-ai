"""Stage 5: FastAPI backend for the IndustrialSense dashboard.

Serves the precomputed IMS replays (dashboard/data, built by
scripts/build_dashboard_data.py) and the built React app (dashboard/web/dist).

Run:  python -m uvicorn api:app --app-dir dashboard --port 8765
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DIST = HERE / "web" / "dist"

app = FastAPI(title="IndustrialSense AI", version="0.5.0")


@lru_cache(maxsize=None)
def _read(name: str) -> dict | list:
    return json.loads((DATA / name).read_text())


@app.get("/api/tests")
def tests() -> list:
    """The three IMS run-to-failure tests available for replay."""
    return _read("index.json")


@app.get("/api/tests/{test}")
def test(test: str) -> dict:
    """One test: snapshot times, and per bearing the RMS, the drift-following
    threshold, the alarm state and an explanation for every alarm onset."""
    names = {t["test"] for t in _read("index.json")}
    if test not in names:
        raise HTTPException(404, f"unknown test {test!r}; available: {sorted(names)}")
    return _read(f"ims_{test}.json")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "tests": len(_read("index.json")), "frontend_built": DIST.exists()}


if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="web")
