"""FastAPI service: forecast metadata, map layers, point forecasts and plan checks.

Run locally:  uvicorn rainwin.api:app --reload  (from backend/)
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .insights import chance_curve, dry_spells, plan_check
from .store import Run, Store

log = logging.getLogger("rainwin.api")

store = Store(config.DATA_DIR)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Optional in-process scheduler for single-container deploys.
    if os.environ.get("RAINWIN_AUTO_REFRESH") == "1":
        interval = int(os.environ.get("RAINWIN_REFRESH_SECONDS", "1800"))
        threading.Thread(target=_refresh_loop, args=(interval,), daemon=True).start()
    yield


app = FastAPI(title="Rain Window API", version="1.0.0", lifespan=lifespan,
              description="ECMWF IFS precipitation forecasts for India, with ensemble-based rain chances.")
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("RAINWIN_CORS", "*").split(","),
    allow_methods=["GET"],
    allow_headers=["*"],
)

PUBLIC_FILES = {"manifest.json", "hres_rate.bin", "ens_prob.bin"}

# Shared, validated query parameters (rejects inf/nan and out-of-range values with 422).
Lat = Annotated[float, Query(ge=-90, le=90, description="Latitude, degrees north")]
Lon = Annotated[float, Query(ge=-180, le=180, description="Longitude, degrees east")]
Threshold = Annotated[float, Query(gt=0, le=100, description="Rain amount that counts as wet, mm")]
DEFAULT_THRESHOLD_MM = 1.0
AREA_RADIUS_CELLS = 1  # neighbourhood = 3 x 3 cells around the point


def _run() -> Run:
    run = store.get()
    if run is None:
        raise HTTPException(503, "No forecast published yet; the pipeline is still running.")
    return run


def _point_or_400(run: Run, lat: float, lon: float):
    try:
        return run.point(lat, lon)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def _parse_time(value: str) -> datetime:
    try:
        t = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(400, f"invalid datetime {value!r}") from exc
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def _r(a, nd: int = 2) -> list:
    """Round for JSON without float32 artefacts (0.0099999 -> 0.01)."""
    return np.round(np.asarray(a, dtype=np.float64), nd).tolist()


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).isoformat(timespec="minutes").replace("+00:00", "Z")


@app.get("/api/health")
def health():
    run = store.get()
    return {"status": "ok" if run else "warming_up", "run_id": run.run_id if run else None}


@app.get("/api/manifest")
def manifest():
    run = _run()
    return {**run.manifest, "base_url": f"/api/runs/{run.run_id}/"}


@app.get("/api/runs/{run_id}/{name}")
def run_file(run_id: str, name: str):
    if name not in PUBLIC_FILES or not run_id.isdigit():
        raise HTTPException(404)
    path = store.data_dir / "runs" / run_id / name
    if not path.exists():
        raise HTTPException(404, "run not available (it may have been pruned)")
    # run files never change once published
    return FileResponse(path, headers={"Cache-Control": "public, max-age=31536000, immutable"},
                        media_type="application/json" if name.endswith(".json") else "application/octet-stream")


@app.get("/api/point")
def point(lat: Lat, lon: Lon, members: bool = False):
    """Time series at the nearest grid cell: HRES, ensemble percentiles and probabilities."""
    run = _run()
    j, i, hres, mem = _point_or_400(run, lat, lon)
    pct = np.percentile(mem, [10, 50, 90], axis=0)
    out = {
        "run_id": run.run_id,
        "grid_lat": float(run.grid.lats[j]),
        "grid_lon": float(run.grid.lons[i]),
        "frames": run.manifest["frames"],
        "hres": _r(hres),
        "ens": {
            "p10": _r(pct[0]),
            "p50": _r(pct[1]),
            "p90": _r(pct[2]),
            "mean": _r(mem.mean(axis=0)),
            "prob": {str(t): np.round((mem >= t).mean(axis=0) * 100).astype(int).tolist()
                     for t in config.THRESHOLDS_MMH},
        },
    }
    if members:
        out["members"] = _r(mem)
    return out


@app.get("/api/plan")
def plan(lat: Lat, lon: Lon, start: str, end: str,
         threshold_mm: Threshold = DEFAULT_THRESHOLD_MM,
         now: Annotated[str | None, Query(description="Override the current time (testing)")] = None):
    """Will it rain on my plans? Chance of >= threshold_mm between start and end.

    A window that has already ended is rejected. A window already under way is
    evaluated from now until its end (``clipped`` is true and ``start`` is now).
    ``area_chance`` is the chance of that much rain anywhere within one grid cell
    (about 28 km) of the point.
    """
    run = _run()
    t0, t1 = _parse_time(start), _parse_time(end)
    t_now = _parse_time(now) if now else datetime.now(timezone.utc)
    if t1 <= t0:
        raise HTTPException(400, "end must be after start")
    if t1 <= t_now:
        raise HTTPException(400, "that window has already passed")
    clipped = t0 < t_now
    if clipped:
        t0 = t_now
    a, b = run.hours_since_run(t0), run.hours_since_run(t1)
    if b > run.steps[-1] or a < run.steps[0]:
        raise HTTPException(400, "window is outside the forecast range")
    if b - a > 72:
        raise HTTPException(400, "window longer than 72 hours")
    j, i, hres, mem = _point_or_400(run, lat, lon)
    return {
        "run_id": run.run_id,
        "start": _iso(t0), "end": _iso(t1), "clipped": clipped, "threshold_mm": threshold_mm,
        "grid_lat": float(run.grid.lats[j]), "grid_lon": float(run.grid.lons[i]),
        "area_radius_km": round(AREA_RADIUS_CELLS * run.grid.dlat * 111.2),
        **plan_check(hres, mem, run.steps, a, b, threshold_mm,
                     area_members=run.area(j, i, AREA_RADIUS_CELLS)),
    }


@app.get("/api/dry-windows")
def dry_windows(lat: Lat, lon: Lon,
                duration_h: float = Query(2, gt=0, le=24),
                within_h: float = Query(48, gt=0, le=240),
                threshold_mm: Threshold = DEFAULT_THRESHOLD_MM,
                start: str | None = None,
                earliest_hour: int = Query(6, ge=0, le=23),
                latest_hour: int = Query(22, ge=1, le=24),
                tz_offset_min: int = Query(330, ge=-720, le=840)):
    """Hour-by-hour rain chance for an activity, and the best dry spells.

    ``curve`` gives the chance for a ``duration_h`` window starting each hour.
    ``best`` lists up to 3 non-overlapping dry spells: back-to-back dry slots are
    merged, so a spell can be longer than ``duration_h``; its ``chance`` is the
    worst ``duration_h`` window inside it. Spells start at or after
    ``earliest_hour`` and finish by ``latest_hour`` local time (default IST, 06-22).
    """
    run = _run()
    now = _parse_time(start) if start else datetime.now(timezone.utc)
    origin = max(run.hours_since_run(now), float(run.steps[0]))
    origin = float(np.ceil(origin))  # whole hours after the run start
    last = min(origin + within_h, float(run.steps[-1])) - duration_h
    if last < origin:
        raise HTTPException(400, "not enough forecast left for that duration")
    starts = np.arange(origin, last + 1e-9, 1.0)
    _, _, _, mem = _point_or_400(run, lat, lon)
    curve = chance_curve(mem, run.steps, starts, duration_h, threshold_mm)

    def at(h):
        return _iso(run.run_time + timedelta(hours=float(h)))

    tz = timezone(timedelta(minutes=tz_offset_min))
    local_start = [(run.run_time + timedelta(hours=float(h))).astimezone(tz) for h in starts]
    allowed = np.array([
        t.hour >= earliest_hour and t.hour + t.minute / 60 + duration_h <= latest_hour
        for t in local_start
    ])
    spells = dry_spells(curve, starts, allowed, duration_h)

    return {
        "run_id": run.run_id, "duration_h": duration_h, "threshold_mm": threshold_mm,
        "curve": [{"start": at(s), "chance": round(float(c))} for s, c in zip(starts, curve)],
        "best": [{"start": at(sp["start_h"]), "end": at(sp["end_h"]),
                  "hours": sp["end_h"] - sp["start_h"], "chance": round(sp["chance"])} for sp in spells],
    }


def _refresh_loop(interval_s: int) -> None:
    while True:
        try:
            log.info("refresh: running pipeline")
            subprocess.run([sys.executable, "-m", "rainwin.build", "--source",
                            os.environ.get("RAINWIN_SOURCE", "ecmwf")],
                           check=True, cwd=Path(__file__).resolve().parents[1])
        except Exception as exc:
            log.error("refresh failed: %s", exc)
        time.sleep(interval_s)


# Serve the built frontend from the same origin when present (single-container deploy).
_WEB_DIST = Path(os.environ.get("RAINWIN_WEB_DIST", Path(__file__).resolve().parents[2] / "web" / "dist"))
if _WEB_DIST.is_dir():
    app.mount("/", StaticFiles(directory=_WEB_DIST, html=True), name="web")
