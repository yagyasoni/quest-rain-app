"""Pipeline entry point: fetch the latest IFS run and write web-ready products.

Usage:
    python -m rainwin.build [--source ecmwf|aws|azure|google] [--max-step 240]

Output layout (under DATA_DIR):
    latest.json                      -> {"run_id": ...}
    runs/<run_id>/manifest.json      metadata: grid, steps, valid times, files
    runs/<run_id>/hres_rate.bin      uint16 [t, y, x]  HRES mean rate, mm/h * SCALE
    runs/<run_id>/ens_prob.bin       uint8  [k, t, y, x] % of members >= threshold k
    runs/<run_id>/ens_members.npy    uint16 [m, t, y, x]  per-member rates (API only)
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import config
from .fetch import download, latest_complete_run, read_cropped
from .process import deaccumulate, decode_u16, encode_u16, exceedance_probability

log = logging.getLogger("rainwin.build")


def _retry(fn, *args, attempts: int = 4, **kwargs):
    for i in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # network hiccups, partial publication
            if i == attempts - 1:
                raise
            wait = 5 * 2 ** i
            log.warning("attempt %d failed (%s); retrying in %ds", i + 1, exc, wait)
            time.sleep(wait)


def build(source: str = "ecmwf", max_step: int = config.MAX_STEP, workers: int = 4,
          data_dir: Path = config.DATA_DIR, run: datetime | None = None,
          force: bool = False) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    run = run or latest_complete_run(source, max_step)
    run_id = run.strftime("%Y%m%d%H")
    final = data_dir / "runs" / run_id
    if final.exists() and not force:
        log.info("run %s already built; nothing to do", run_id)
        _write_latest(data_dir, run_id)
        return final

    steps = config.common_steps(run.hour, max_step)  # shared by HRES and ENS
    log.info("building run %s (%d steps to +%dh) from %s", run_id, len(steps), steps[-1], source)

    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="rainwin-") as tmp:
        tmp = Path(tmp)

        # --- HRES: one request for all steps (~21 MB) --------------------------
        _retry(download, source, run, "oper", steps, tmp / "hres.grib2")
        grid, fields = read_cropped(tmp / "hres.grib2", config.BBOX)
        missing = [s for s in steps if (0, s) not in fields]
        if missing:
            raise RuntimeError(f"HRES missing steps {missing}")
        hres_cum = np.stack([fields[(0, s)] for s in steps])  # [t+1, y, x]
        hres_rate = deaccumulate(hres_cum, steps, axis=0)       # [t, y, x]
        log.info("HRES done in %.0fs, grid %dx%d", time.time() - t0, grid.ny, grid.nx)

        # --- ENS: one request per step, cropped immediately (~21 MB each) -----
        # Results are consumed in step order and de-accumulated on the fly into a
        # uint16 cube, so peak memory stays ~150 MB instead of several float32 copies.
        def ens_step(step: int) -> np.ndarray:
            path = tmp / f"ens_{step:03d}.grib2"
            _retry(download, source, run, "enfo", step, path)
            g, f = read_cropped(path, config.BBOX)
            path.unlink()
            if g != grid:
                raise RuntimeError("ENS grid differs from HRES grid")
            numbers = sorted(n for (n, s) in f if s == step)
            if len(numbers) != 50:
                raise RuntimeError(f"ENS step {step}: expected 50 members, got {len(numbers)}")
            return np.stack([f[(n, step)] for n in numbers])  # [m, y, x], metres accumulated

        n_frames = len(steps) - 1
        members = np.empty((50, n_frames, grid.ny, grid.nx), dtype="<u2")
        prev = None
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for k, cum in enumerate(pool.map(ens_step, steps)):
                if prev is not None:
                    rate = deaccumulate(np.stack([prev, cum], axis=1), steps[k - 1:k + 1], axis=1)
                    members[:, k - 1] = encode_u16(rate[:, 0])
                prev = cum
        prob = np.empty((len(config.THRESHOLDS_MMH), n_frames, grid.ny, grid.nx), dtype=np.uint8)
        for k in range(n_frames):
            prob[:, k] = exceedance_probability(decode_u16(members[:, k]), config.THRESHOLDS_MMH)
        log.info("ENS done in %.0fs", time.time() - t0)

        # --- write products ---------------------------------------------------
        out = data_dir / "runs" / f".{run_id}.partial"
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        encode_u16(hres_rate).tofile(out / "hres_rate.bin")
        prob.tofile(out / "ens_prob.bin")
        np.save(out / "ens_members.npy", members)

        manifest = _manifest(run, run_id, steps, grid, members.shape[0], hres_rate, source)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1))

        shutil.rmtree(final, ignore_errors=True)
        out.rename(final)

    _write_latest(data_dir, run_id)
    _prune(data_dir)
    log.info("run %s published in %.0fs", run_id, time.time() - t0)
    return final


def _manifest(run, run_id, steps, grid, n_members, hres_rate, source) -> dict:
    run_utc = run.replace(tzinfo=timezone.utc)
    frames = [
        {
            "start": (run_utc + timedelta(hours=a)).isoformat().replace("+00:00", "Z"),
            "end": (run_utc + timedelta(hours=b)).isoformat().replace("+00:00", "Z"),
            "hours": b - a,
        }
        for a, b in zip(steps[:-1], steps[1:])
    ]
    return {
        "run_id": run_id,
        "run_time": run_utc.isoformat().replace("+00:00", "Z"),
        "cycle": f"{run.hour:02d}",
        "horizon_h": steps[-1],
        "model": "ECMWF IFS (HRES + 50-member ENS), 0.25°",
        "source": source,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "grid": grid.to_dict(),
        "steps": steps,
        "frames": frames,
        "n_members": n_members,
        "scale": config.SCALE,
        "thresholds_mmh": config.THRESHOLDS_MMH,
        "focus": config.FOCUS,
        "max_rate_mmh": float(np.round(hres_rate.max(), 2)),
        "files": {
            "hres_rate": {"path": "hres_rate.bin", "dtype": "uint16", "shape": [len(frames), grid.ny, grid.nx]},
            "ens_prob": {"path": "ens_prob.bin", "dtype": "uint8",
                         "shape": [len(config.THRESHOLDS_MMH), len(frames), grid.ny, grid.nx]},
        },
        "attribution": config.ATTRIBUTION,
    }


def _write_latest(data_dir: Path, run_id: str) -> None:
    tmp = data_dir / ".latest.json.tmp"
    tmp.write_text(json.dumps({"run_id": run_id}))
    tmp.replace(data_dir / "latest.json")


def _prune(data_dir: Path) -> None:
    runs = sorted(p for p in (data_dir / "runs").iterdir() if p.is_dir() and not p.name.startswith("."))
    for old in runs[: -config.KEEP_RUNS]:
        shutil.rmtree(old, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="ecmwf", help="ecmwf | aws | azure | google")
    ap.add_argument("--max-step", type=int, default=config.MAX_STEP)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--run", help="specific run as YYYYMMDDHH (default: freshest complete run)")
    ap.add_argument("--force", action="store_true", help="rebuild even if the run exists")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run = datetime.strptime(args.run, "%Y%m%d%H") if args.run else None
    build(args.source, args.max_step, args.workers, run=run, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
