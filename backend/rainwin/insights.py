"""Turn ensemble numbers into decisions: window rain chance, dry windows, wording."""
from __future__ import annotations

import numpy as np

from .process import window_totals


def plan_check(hres: np.ndarray, members: np.ndarray, steps: np.ndarray,
               start_h: float, end_h: float, threshold_mm: float,
               area_members: np.ndarray | None = None) -> dict:
    """Chance that at least ``threshold_mm`` falls between start_h and end_h.

    ``members`` is the [m, t] series at the nearest grid cell. ``area_members``
    ([m, t, cy, cx], the surrounding block of cells) adds a neighbourhood chance:
    the share of members that bring the threshold to *any* cell in the block.
    Showers are hit-or-miss at 28 km, so this is the "rain nearby" number.
    """
    totals = window_totals(members, steps, start_h, end_h)
    wet = totals >= threshold_mm
    chance = float(wet.mean() * 100.0)
    wet_totals = totals[wet]
    out = {
        "chance": round(chance),
        "members_wet": wet.tolist(),
        "member_totals_mm": np.round(totals, 2).tolist(),
        "hres_total_mm": round(float(window_totals(hres, steps, start_h, end_h)), 2),
        "median_total_mm": round(float(np.median(totals)), 2),
        "p90_total_mm": round(float(np.percentile(totals, 90)), 2),
        "median_if_wet_mm": round(float(np.median(wet_totals)), 2) if wet_totals.size else 0.0,
        "verdict": verdict(chance, float(np.median(wet_totals)) if wet_totals.size else 0.0),
    }
    if area_members is not None:
        # [m, t, cy, cx] -> per-cell window totals [m, cy, cx]
        cell_totals = window_totals(np.moveaxis(area_members, 1, -1), steps, start_h, end_h)
        area_wet = (cell_totals >= threshold_mm).reshape(cell_totals.shape[0], -1).any(axis=1)
        out["area_chance"] = round(float(area_wet.mean() * 100.0))
    return out


def chance_curve(members: np.ndarray, steps: np.ndarray, starts_h: np.ndarray,
                 duration_h: float, threshold_mm: float) -> np.ndarray:
    """Rain chance (%) for a window of ``duration_h`` starting at each of ``starts_h``."""
    lo, hi = steps[:-1], steps[1:]
    # overlap matrix [n_starts, n_intervals] -> totals [m, n_starts]
    a = starts_h[:, None]
    overlap = np.clip(np.minimum(hi, a + duration_h) - np.maximum(lo, a), 0.0, None)
    totals = members @ overlap.T
    return (totals >= threshold_mm).mean(axis=0) * 100.0


def dry_spells(curve: np.ndarray, starts_h: np.ndarray, allowed: np.ndarray,
               duration_h: float, k: int = 3, tolerance: float = 5.0) -> list[dict]:
    """Group the driest window starts into longer dry spells.

    ``curve[i]`` is the rain chance for a ``duration_h`` window starting at
    ``starts_h[i]`` (hourly). Starting from the driest allowed start, a spell
    grows over neighbouring hourly starts whose chance is within ``tolerance``
    percentage points of the best, so back-to-back dry slots come out as one
    spell. Spells never overlap. Returned chance is the worst window inside it.
    Ties are broken by earliest start.
    """
    n = len(starts_h)
    free = np.asarray(allowed, dtype=bool).copy()
    spells: list[dict] = []
    while len(spells) < k and free.any():
        cand = np.where(free)[0]
        best = int(cand[np.lexsort((starts_h[cand], np.round(curve[cand], 6)))[0]])
        limit = curve[best] + tolerance

        def joins(i: int, neighbour: int) -> bool:
            """Can start ``i`` extend the spell that currently ends at ``neighbour``?"""
            return (0 <= i < n and free[i] and curve[i] <= limit
                    and abs(abs(starts_h[i] - starts_h[neighbour]) - 1.0) < 1e-9)

        lo = hi = best
        while joins(lo - 1, lo):
            lo -= 1
        while joins(hi + 1, hi):
            hi += 1
        start, end = float(starts_h[lo]), float(starts_h[hi] + duration_h)
        spells.append({"start_h": start, "end_h": end, "chance": float(curve[lo:hi + 1].max())})
        # later spells must not overlap this one: drop windows touching [start, end)
        free &= ~((starts_h < end) & (starts_h + duration_h > start))
    return spells


def verdict(chance: float, wet_amount_mm: float) -> str:
    if chance < 10:
        return "Almost certainly dry"
    if chance < 30:
        head = "Probably dry"
    elif chance < 60:
        head = "Could go either way"
    elif chance < 85:
        head = "Rain likely"
    else:
        head = "Rain almost certain"
    if wet_amount_mm < 1:
        kind = "a light drizzle at most"
    elif wet_amount_mm < 5:
        kind = "a proper shower"
    elif wet_amount_mm < 20:
        kind = "heavy rain"
    else:
        kind = "a downpour"
    return f"{head}: expect {kind}" if chance >= 60 else f"{head}. If it does rain, expect {kind}"
