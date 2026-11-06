"""Static configuration for the Rain Window pipeline and API."""
from __future__ import annotations

import os
from pathlib import Path

# Region of interest: the Indian subcontinent, with Bengaluru as the focus city.
BBOX = {"north": 38.0, "south": 5.0, "west": 66.0, "east": 100.0}
FOCUS = {"name": "Bengaluru", "lat": 12.9716, "lon": 77.5946}

# Precipitation rates (mm/h, averaged over each forecast interval) used for
# ensemble exceedance probabilities on the map.
THRESHOLDS_MMH = [0.1, 1.0, 4.0]

# uint16 encoding: stored value = rate_mm_per_h * SCALE  (max ~655 mm/h)
SCALE = 100

# Maximum forecast horizon we process (hours). We always use the freshest run:
# 00/12 UTC runs reach 240 h (HRES) / 360 h (ENS); 06/18 UTC runs reach 90 h
# (HRES) / 144 h (ENS). We process the steps both share, capped at MAX_STEP,
# so the HRES and ENS layers line up: 240 h for 00/12 runs, 90 h for 06/18.
MAX_STEP = int(os.environ.get("RAINWIN_MAX_STEP", "240"))

DATA_DIR = Path(os.environ.get("RAINWIN_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))
KEEP_RUNS = int(os.environ.get("RAINWIN_KEEP_RUNS", "2"))

ATTRIBUTION = (
    "Contains modified ECMWF Open Data (IFS), © ECMWF, licensed under CC BY 4.0. "
    "ECMWF does not accept liability for this product."
)


def steps_for(stream: str, cycle_hour: int, max_step: int = MAX_STEP) -> list[int]:
    """Forecast steps (hours) published in ECMWF open data for a stream/cycle."""
    main = cycle_hour in (0, 12)
    if stream == "oper":
        steps = list(range(0, 145, 3)) + list(range(150, 241, 6)) if main else list(range(0, 91, 3))
    elif stream == "enfo":
        steps = list(range(0, 145, 3)) + list(range(150, 361, 6)) if main else list(range(0, 145, 3))
    else:
        raise ValueError(f"unknown stream {stream!r}")
    return [s for s in steps if s <= max_step]


def common_steps(cycle_hour: int, max_step: int = MAX_STEP) -> list[int]:
    """Steps available in both HRES (oper) and ENS (enfo) for this cycle."""
    ens = set(steps_for("enfo", cycle_hour, max_step))
    return [s for s in steps_for("oper", cycle_hour, max_step) if s in ens]
