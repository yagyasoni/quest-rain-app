"""API tests against a tiny synthetic run with known answers."""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rainwin import api
from rainwin.process import encode_u16
from rainwin.store import Store

STEPS = [0, 3, 6, 12]          # 3 frames: 3h, 3h, 6h
RUN = "2026110600"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    run_dir = tmp_path / "runs" / RUN
    run_dir.mkdir(parents=True)
    ny, nx, m = 3, 4, 10
    hres = np.zeros((3, ny, nx), np.float32)
    hres[1] = 2.0                                     # 2 mm/h during 03-06Z
    members = np.zeros((m, 3, ny, nx), np.float32)
    members[:4, 1] = 1.0                              # 4 of 10 members wet 03-06Z everywhere
    members[4, 1, 1, 2] = 2.0                         # member 5 wet only in the cell north of the point
    encode_u16(hres).tofile(run_dir / "hres_rate.bin")
    np.save(run_dir / "ens_members.npy", encode_u16(members))
    (run_dir / "ens_prob.bin").write_bytes(b"\0")
    manifest = {
        "run_id": RUN, "run_time": "2026-11-06T00:00:00Z", "steps": STEPS,
        "frames": [{"start": "x", "end": "y", "hours": h} for h in (3, 3, 6)],
        "grid": {"lat0": 13.5, "lon0": 77.0, "dlat": 0.25, "dlon": 0.25, "ny": ny, "nx": nx},
        "files": {"hres_rate": {"shape": [3, ny, nx]}},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "latest.json").write_text(json.dumps({"run_id": RUN}))
    monkeypatch.setattr(api, "store", Store(tmp_path))
    return TestClient(api.app)


def test_health_and_manifest(client):
    assert client.get("/api/health").json() == {"status": "ok", "run_id": RUN}
    m = client.get("/api/manifest").json()
    assert m["base_url"] == f"/api/runs/{RUN}/"


def test_run_files_whitelisted(client):
    r = client.get(f"/api/runs/{RUN}/hres_rate.bin")
    assert r.status_code == 200 and "immutable" in r.headers["cache-control"]
    assert client.get(f"/api/runs/{RUN}/ens_members.npy").status_code == 404
    assert client.get("/api/runs/../latest.json").status_code == 404


def test_point_series(client):
    r = client.get("/api/point", params={"lat": 13.0, "lon": 77.5, "members": "true"}).json()
    assert (r["grid_lat"], r["grid_lon"]) == (13.0, 77.5)
    assert r["hres"] == [0.0, 2.0, 0.0]
    assert r["ens"]["prob"]["0.1"] == [0, 40, 0]
    assert len(r["members"]) == 10


def test_point_outside_domain(client):
    assert client.get("/api/point", params={"lat": 40, "lon": 0}).status_code == 400


NOW = "2026-11-06T00:00:00Z"   # fixed "now" so the fixture windows are in the future
PT = {"lat": 13, "lon": 77.5}


def plan(client, **kw):
    return client.get("/api/plan", params={**PT, "now": NOW, **kw})


def test_plan_partial_window(client):
    # 04:30-06:00 UTC overlaps 1.5 h of the 03-06Z interval: wet members get 1.5 mm
    r = plan(client, start="2026-11-06T04:30:00Z", end="2026-11-06T06:00:00Z", threshold_mm=1.0).json()
    assert r["chance"] == 40 and r["hres_total_mm"] == 3.0 and r["clipped"] is False
    assert r["members_wet"][:4] == [True] * 4 and not any(r["members_wet"][4:])
    # timezone-aware input (IST) is converted correctly: 10:00-11:30 IST = 04:30-06:00 UTC
    r2 = plan(client, start="2026-11-06T10:00:00+05:30", end="2026-11-06T11:30:00+05:30", threshold_mm=1.0).json()
    assert r2["chance"] == 40
    # threshold above what point members produce -> dry
    r3 = plan(client, start="2026-11-06T04:30:00Z", end="2026-11-06T06:00:00Z", threshold_mm=2.0).json()
    assert r3["chance"] == 0 and r3["verdict"] == "Almost certainly dry"


def test_plan_area_chance_counts_rain_nearby(client):
    r = plan(client, start="2026-11-06T04:30:00Z", end="2026-11-06T06:00:00Z", threshold_mm=1.0).json()
    # member 5 rains only one cell away: not at the point, but within the area
    assert r["chance"] == 40 and r["area_chance"] == 50 and r["area_radius_km"] == 28
    # 2 mm threshold: only member 5's neighbouring cell (2 mm/h * 1.5 h = 3 mm) qualifies
    r2 = plan(client, start="2026-11-06T04:30:00Z", end="2026-11-06T06:00:00Z", threshold_mm=2.0).json()
    assert r2["chance"] == 0 and r2["area_chance"] == 10


def test_plan_default_threshold_is_1mm(client):
    r = plan(client, start="2026-11-06T04:30:00Z", end="2026-11-06T06:00:00Z").json()
    assert r["threshold_mm"] == 1.0


def test_plan_past_and_ongoing_windows(client):
    # fully in the past -> rejected
    r = client.get("/api/plan", params={**PT, "now": "2026-11-06T08:00:00Z",
                                        "start": "2026-11-06T03:00:00Z", "end": "2026-11-06T06:00:00Z"})
    assert r.status_code == 400 and "already passed" in r.json()["detail"]
    # under way: 03:00-06:00Z asked at 05:00Z -> only 05:00-06:00Z (1 mm for wet members) is evaluated
    r = client.get("/api/plan", params={**PT, "now": "2026-11-06T05:00:00Z", "threshold_mm": 1.0,
                                        "start": "2026-11-06T03:00:00Z", "end": "2026-11-06T06:00:00Z"}).json()
    assert r["clipped"] is True and r["start"] == "2026-11-06T05:00Z"
    assert r["member_totals_mm"][:4] == [1.0] * 4 and r["chance"] == 40


def test_plan_validation(client):
    bad = [
        {"start": "2026-11-06T06:00Z", "end": "2026-11-06T05:00Z"},   # reversed
        {"start": "2026-11-06T06:00Z", "end": "2026-11-07T06:00Z"},   # beyond 12 h fixture run
        {"start": "nonsense", "end": "2026-11-06T05:00Z"},
    ]
    for q in bad:
        assert plan(client, **q).status_code == 400


def test_non_finite_and_out_of_range_coordinates_are_rejected(client):
    win = {"start": "2026-11-06T04:00Z", "end": "2026-11-06T05:00Z", "now": NOW}
    for lat in ("inf", "-inf", "nan", "91"):
        assert client.get("/api/plan", params={"lat": lat, "lon": 77.5, **win}).status_code == 422
        assert client.get("/api/dry-windows", params={"lat": lat, "lon": 77.5}).status_code == 422
        assert client.get("/api/point", params={"lat": lat, "lon": 77.5}).status_code == 422


def test_dry_windows_merges_back_to_back_slots_into_spells(client):
    r = client.get("/api/dry-windows", params={"lat": 13, "lon": 77.5, "duration_h": 2,
                                               "start": "2026-11-06T00:00:00Z",
                                               "earliest_hour": 0, "latest_hour": 24}).json()
    chances = {c["start"]: c["chance"] for c in r["curve"]}
    assert chances["2026-11-06T00:00Z"] == 0       # 00-02Z fully dry
    assert chances["2026-11-06T03:00Z"] == 40      # 03-05Z inside the wet interval (2 mm >= 1)
    assert r["best"] == [
        {"start": "2026-11-06T00:00Z", "end": "2026-11-06T03:00Z", "hours": 3.0, "chance": 0},
        {"start": "2026-11-06T06:00Z", "end": "2026-11-06T12:00Z", "hours": 6.0, "chance": 0},
        {"start": "2026-11-06T03:00Z", "end": "2026-11-06T06:00Z", "hours": 3.0, "chance": 40},
    ]


def test_dry_windows_respects_local_waking_hours(client):
    # 00Z = 05:30 IST: with the default 06:00 IST start, the first spell begins at 01Z
    r = client.get("/api/dry-windows", params={"lat": 13, "lon": 77.5, "duration_h": 2,
                                               "start": "2026-11-06T00:00:00Z"}).json()
    first = r["best"][0]
    assert (first["start"], first["end"], first["chance"]) == ("2026-11-06T01:00Z", "2026-11-06T03:00Z", 0)
    assert all(b["start"] != "2026-11-06T00:00Z" for b in r["best"])
