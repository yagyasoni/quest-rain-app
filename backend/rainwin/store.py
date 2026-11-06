"""Read-side access to the latest published run (used by the API)."""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .process import Grid, decode_u16


@dataclass
class Run:
    run_id: str
    path: Path
    manifest: dict
    grid: Grid
    run_time: datetime
    steps: np.ndarray        # hours since run, len = n_frames + 1
    hres: np.ndarray         # float32 [t, y, x] mm/h
    members: np.ndarray      # uint16 memmap [m, t, y, x] (mm/h * scale)

    def hours_since_run(self, when: datetime) -> float:
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return (when - self.run_time).total_seconds() / 3600.0

    def point(self, lat: float, lon: float) -> tuple[int, int, np.ndarray, np.ndarray]:
        """Nearest-cell HRES series [t] and member series [m, t] in mm/h."""
        j, i = self.grid.nearest(lat, lon)
        hres = self.hres[:, j, i]
        members = decode_u16(np.asarray(self.members[:, :, j, i]))
        return j, i, hres, members

    def area(self, j: int, i: int, radius: int = 1) -> np.ndarray:
        """Member series for the block of cells within ``radius`` of (j, i): [m, t, cy, cx] mm/h.

        The block is clipped at the domain edge.
        """
        j0, j1 = max(0, j - radius), min(self.grid.ny, j + radius + 1)
        i0, i1 = max(0, i - radius), min(self.grid.nx, i + radius + 1)
        return decode_u16(np.asarray(self.members[:, :, j0:j1, i0:i1]))


class Store:
    """Caches the latest run; reloads when ``latest.json`` points elsewhere."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self._run: Run | None = None
        self._lock = threading.Lock()

    def latest_id(self) -> str | None:
        try:
            return json.loads((self.data_dir / "latest.json").read_text())["run_id"]
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            return None

    def get(self) -> Run | None:
        run_id = self.latest_id()
        if run_id is None:
            return None
        if self._run is not None and self._run.run_id == run_id:
            return self._run
        with self._lock:
            if self._run is None or self._run.run_id != run_id:
                self._run = self._load(run_id)
        return self._run

    def _load(self, run_id: str) -> Run:
        path = self.data_dir / "runs" / run_id
        manifest = json.loads((path / "manifest.json").read_text())
        g = manifest["grid"]
        grid = Grid(g["lat0"], g["lon0"], g["dlat"], g["dlon"], g["ny"], g["nx"])
        shape = manifest["files"]["hres_rate"]["shape"]
        hres = decode_u16(np.fromfile(path / "hres_rate.bin", dtype="<u2").reshape(shape))
        members = np.load(path / "ens_members.npy", mmap_mode="r")
        run_time = datetime.fromisoformat(manifest["run_time"].replace("Z", "+00:00"))
        return Run(run_id, path, manifest, grid, run_time,
                   np.asarray(manifest["steps"], dtype=np.float64), hres, members)
