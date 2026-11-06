"""Download IFS precipitation from ECMWF Open Data and read it into cropped arrays."""
from __future__ import annotations

import functools
import logging
import os
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import eccodes
import ecmwf.opendata.client as _ecmwf_client
import numpy as np
from ecmwf.opendata import Client
from multiurl.base import NoBar

from .config import common_steps
from .process import Grid, crop_plan

log = logging.getLogger(__name__)

# multiurl hard-codes tqdm progress bars (disable=False); they flood CI logs.
if os.environ.get("RAINWIN_PROGRESS") != "1":
    _ecmwf_client.download = functools.partial(_ecmwf_client.download, progress_bar=NoBar)


def client(source: str) -> Client:
    return Client(source=source, model="ifs", preserve_request_order=False)


def freshest_complete_run(latest: Callable[[str, int, int], datetime | None], max_step: int) -> datetime:
    """Newest run (any of the 00/06/12/18 UTC cycles) whose HRES and ENS are both
    published up to the last step we process for that cycle.

    ``latest(stream, cycle_hour, step)`` returns the newest run of that cycle that
    has ``step`` published, or None. Kept injectable so the choice can be tested.
    """
    candidates = []
    for hour in (0, 6, 12, 18):
        last = common_steps(hour, max_step)[-1]
        hres, ens = latest("oper", hour, last), latest("enfo", hour, last)
        if hres is not None and ens is not None:
            candidates.append(min(hres, ens))  # newest run of this cycle with both complete
    if not candidates:
        raise RuntimeError("no complete IFS run found in the last two days")
    return max(candidates)


def latest_complete_run(source: str, max_step: int) -> datetime:
    c = client(source)

    def latest(stream: str, hour: int, step: int) -> datetime | None:
        try:
            return c.latest(stream=stream, type="fc" if stream == "oper" else "pf",
                            param="tp", time=hour, step=step)
        except ValueError:  # nothing for this cycle in the last two days
            return None

    return freshest_complete_run(latest, max_step)


def download(source: str, run: datetime, stream: str, steps: list[int] | int, target: Path) -> Path:
    typ = "fc" if stream == "oper" else "pf"
    client(source).retrieve(
        date=run.strftime("%Y%m%d"), time=run.hour, stream=stream, type=typ,
        param="tp", step=steps, target=str(target),
    )
    return target


def read_cropped(path: Path, bbox: dict) -> tuple[Grid, dict[tuple[int, int], np.ndarray]]:
    """Read every tp message in ``path``, cropped to ``bbox``.

    Returns the cropped grid and a mapping ``(member_number, end_step) -> 2-D array``
    (metres, accumulated since run start). HRES messages get member number 0.
    """
    fields: dict[tuple[int, int], np.ndarray] = {}
    grid = rows = cols = None
    with open(path, "rb") as fh:
        while (h := eccodes.codes_grib_new_from_file(fh)) is not None:
            try:
                if eccodes.codes_get(h, "shortName") != "tp":
                    continue
                if eccodes.codes_get(h, "jScansPositively") != 0:
                    raise ValueError("expected north-to-south scanning")
                if grid is None:
                    rows, cols, grid = crop_plan(
                        eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees"),
                        eccodes.codes_get(h, "longitudeOfFirstGridPointInDegrees"),
                        eccodes.codes_get(h, "jDirectionIncrementInDegrees"),
                        eccodes.codes_get(h, "iDirectionIncrementInDegrees"),
                        eccodes.codes_get(h, "Nj"), eccodes.codes_get(h, "Ni"), bbox,
                    )
                    nj, ni = eccodes.codes_get(h, "Nj"), eccodes.codes_get(h, "Ni")
                values = eccodes.codes_get_values(h).reshape(nj, ni)
                number = eccodes.codes_get(h, "number") if eccodes.codes_is_defined(h, "number") else 0
                step = int(eccodes.codes_get(h, "endStep"))
                fields[(int(number or 0), step)] = values[np.ix_(rows, cols)].astype(np.float32)
            finally:
                eccodes.codes_release(h)
    if grid is None:
        raise ValueError(f"no tp messages found in {path}")
    return grid, fields
