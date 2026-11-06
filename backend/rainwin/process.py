"""Pure numerical processing: cropping, de-accumulation, encoding, ensemble stats.

Everything here is side-effect free so it can be unit-tested without GRIB files.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SCALE


@dataclass(frozen=True)
class Grid:
    """A regular lat/lon grid (row 0 = northernmost latitude)."""

    lat0: float  # latitude of row 0
    lon0: float  # longitude of column 0 (in [-180, 180))
    dlat: float  # positive spacing; latitude decreases with row index
    dlon: float
    ny: int
    nx: int

    @property
    def lats(self) -> np.ndarray:
        return self.lat0 - np.arange(self.ny) * self.dlat

    @property
    def lons(self) -> np.ndarray:
        return self.lon0 + np.arange(self.nx) * self.dlon

    def nearest(self, lat: float, lon: float) -> tuple[int, int]:
        """Index of the grid cell nearest to (lat, lon); raises if outside."""
        j = int(round((self.lat0 - lat) / self.dlat))
        i = int(round((lon - self.lon0) / self.dlon))
        if not (0 <= j < self.ny and 0 <= i < self.nx):
            raise ValueError(f"point ({lat}, {lon}) is outside the forecast domain")
        return j, i

    def to_dict(self) -> dict:
        return {
            "lat0": self.lat0, "lon0": self.lon0, "dlat": self.dlat, "dlon": self.dlon,
            "ny": self.ny, "nx": self.nx,
            "north": self.lat0, "south": self.lat0 - (self.ny - 1) * self.dlat,
            "west": self.lon0, "east": self.lon0 + (self.nx - 1) * self.dlon,
        }


def crop_plan(lat_first: float, lon_first: float, dlat: float, dlon: float,
              nj: int, ni: int, bbox: dict) -> tuple[np.ndarray, np.ndarray, Grid]:
    """Row/column indices of a global north-to-south grid that fall in ``bbox``.

    Longitudes are wrapped to [-180, 180) first (ECMWF open data starts at 180°E).
    """
    lats = lat_first - np.arange(nj) * dlat
    lons = ((lon_first + np.arange(ni) * dlon + 180.0) % 360.0) - 180.0
    eps = 1e-6
    rows = np.where((lats <= bbox["north"] + eps) & (lats >= bbox["south"] - eps))[0]
    cols = np.where((lons >= bbox["west"] - eps) & (lons <= bbox["east"] + eps))[0]
    if rows.size == 0 or cols.size == 0:
        raise ValueError("bbox does not intersect the grid")
    # keep columns ordered west -> east even if they straddle the wrap point
    cols = cols[np.argsort(lons[cols], kind="stable")]
    grid = Grid(float(lats[rows[0]]), float(lons[cols[0]]), dlat, dlon, rows.size, cols.size)
    return rows, cols, grid


def deaccumulate(cumulative_m: np.ndarray, steps: list[int] | np.ndarray, axis: int = 0) -> np.ndarray:
    """Convert accumulated precipitation (m since run start) to mean rates (mm/h).

    ``cumulative_m`` has the time dimension on ``axis`` with one entry per step.
    Returns ``len(steps) - 1`` intervals; interval k covers (steps[k], steps[k+1]].
    Small negative increments (packing noise) are clipped to zero.
    """
    steps = np.asarray(steps, dtype=np.float64)
    if cumulative_m.shape[axis] != steps.size:
        raise ValueError("time axis length must match number of steps")
    dt = np.diff(steps)
    if np.any(dt <= 0):
        raise ValueError("steps must be strictly increasing")
    inc_mm = np.clip(np.diff(cumulative_m.astype(np.float64), axis=axis), 0.0, None) * 1000.0
    shape = [1] * cumulative_m.ndim
    shape[axis] = dt.size
    return (inc_mm / dt.reshape(shape)).astype(np.float32)


def encode_u16(rate_mmh: np.ndarray) -> np.ndarray:
    """Quantise rates to little-endian uint16 (value / SCALE = mm/h)."""
    return np.clip(np.rint(rate_mmh * SCALE), 0, 65535).astype("<u2")


def decode_u16(raw: np.ndarray) -> np.ndarray:
    return raw.astype(np.float32) / SCALE


def exceedance_probability(members: np.ndarray, thresholds: list[float]) -> np.ndarray:
    """Percent of members (axis 0) with rate >= each threshold -> uint8 (k, ...)."""
    out = np.empty((len(thresholds),) + members.shape[1:], dtype=np.uint8)
    for k, thr in enumerate(thresholds):
        out[k] = np.rint((members >= thr).mean(axis=0) * 100.0).astype(np.uint8)
    return out


def window_totals(rates_mmh: np.ndarray, steps: list[int] | np.ndarray,
                  start_h: float, end_h: float) -> np.ndarray:
    """Rain total (mm) in [start_h, end_h] hours after the run, along the last axis.

    ``rates_mmh[..., k]`` is the mean rate over (steps[k], steps[k+1]]. Intervals
    partially inside the window are weighted by their overlap, i.e. rain is
    assumed uniform within each forecast interval.
    """
    steps = np.asarray(steps, dtype=np.float64)
    if end_h <= start_h:
        raise ValueError("window end must be after start")
    lo, hi = steps[:-1], steps[1:]
    overlap = np.clip(np.minimum(hi, end_h) - np.maximum(lo, start_h), 0.0, None)
    return rates_mmh @ overlap if rates_mmh.ndim > 1 else float(rates_mmh @ overlap)
