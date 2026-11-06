import numpy as np
import pytest

from rainwin import config
from rainwin.process import (
    Grid, crop_plan, deaccumulate, decode_u16, encode_u16,
    exceedance_probability, window_totals,
)


def test_crop_plan_wraps_longitudes_like_ecmwf_open_data():
    # ECMWF open data: first point at 90N, 180E, 0.25 deg, 1440 x 721
    rows, cols, grid = crop_plan(90.0, 180.0, 0.25, 0.25, 721, 1440, config.BBOX)
    assert grid.lat0 == 38.0 and grid.lon0 == 66.0
    assert grid.ny == 133 and grid.nx == 137
    assert rows[0] == (90 - 38) * 4
    assert cols[0] == (66 + 180) * 4  # lon index 0 is -180


def test_nearest_bengaluru_and_out_of_domain():
    _, _, grid = crop_plan(90.0, 180.0, 0.25, 0.25, 721, 1440, config.BBOX)
    j, i = grid.nearest(12.9716, 77.5946)
    assert grid.lats[j] == 13.0 and grid.lons[i] == 77.5
    with pytest.raises(ValueError):
        grid.nearest(51.5, -0.1)


def test_deaccumulate_handles_mixed_step_lengths_and_noise():
    steps = [0, 3, 6, 12]
    cum_m = np.array([0.0, 0.003, 0.003 - 1e-7, 0.009])  # 3 mm, ~0, 6 mm
    rate = deaccumulate(cum_m, steps)
    np.testing.assert_allclose(rate, [1.0, 0.0, 1.0], atol=1e-4)  # mm/h


def test_deaccumulate_member_axis():
    cum = np.zeros((2, 3, 1, 1))
    cum[:, 1] = 0.006
    cum[:, 2] = 0.006
    rate = deaccumulate(cum, [0, 6, 12], axis=1)
    assert rate.shape == (2, 2, 1, 1)
    np.testing.assert_allclose(rate[:, :, 0, 0], [[1, 0], [1, 0]])


def test_deaccumulate_validates():
    with pytest.raises(ValueError):
        deaccumulate(np.zeros(3), [0, 3])
    with pytest.raises(ValueError):
        deaccumulate(np.zeros(3), [0, 3, 3])


def test_u16_roundtrip_and_clipping():
    r = np.array([0.0, 0.123, 12.345, -1.0, 1e6])
    back = decode_u16(encode_u16(r))
    np.testing.assert_allclose(back[:3], r[:3], atol=0.0051)  # half a quantum
    assert back[3] == 0.0 and back[4] == pytest.approx(655.35)


def test_exceedance_probability():
    members = np.array([0.0, 0.5, 2.0, 5.0]).reshape(4, 1)
    p = exceedance_probability(members, [0.1, 1.0, 4.0])
    assert p[:, 0].tolist() == [75, 50, 25]


def test_window_totals_partial_overlap():
    steps = [0, 3, 6, 12]
    rates = np.array([[1.0, 2.0, 0.5], [0.0, 0.0, 0.0]])  # 2 members
    # window 1.5h..9h: 1.5h*1 + 3h*2 + 3h*0.5 = 9 mm
    np.testing.assert_allclose(window_totals(rates, steps, 1.5, 9.0), [9.0, 0.0])
    assert window_totals(rates[0], steps, 0, 12) == pytest.approx(12.0)
    with pytest.raises(ValueError):
        window_totals(rates, steps, 5, 5)


def test_steps_match_open_data_schedule():
    s = config.steps_for("oper", 0, 240)
    assert s[:3] == [0, 3, 6] and 144 in s and 147 not in s and s[-1] == 240
    assert config.steps_for("enfo", 12, 240) == s
    assert config.steps_for("oper", 6, 240)[-1] == 90
    assert config.steps_for("enfo", 18, 240)[-1] == 144


def test_common_steps_per_cycle():
    assert config.common_steps(0, 240) == config.steps_for("oper", 0, 240)
    six = config.common_steps(6, 240)
    assert six == list(range(0, 91, 3))  # HRES 06/18 stops at 90 h
    assert config.common_steps(12, 24)[-1] == 24  # max_step cap applies


def _fake_latest(available):
    """available: {(stream, hour): newest run complete to the cycle's last step}"""
    return lambda stream, hour, step: available.get((stream, hour))


def test_freshest_run_prefers_newest_complete_cycle():
    from datetime import datetime
    from rainwin.fetch import freshest_complete_run
    d = datetime
    avail = {
        ("oper", 0): d(2026, 11, 6, 0), ("enfo", 0): d(2026, 11, 6, 0),
        ("oper", 6): d(2026, 11, 6, 6), ("enfo", 6): d(2026, 11, 6, 6),
        ("oper", 12): d(2026, 11, 5, 12), ("enfo", 12): d(2026, 11, 5, 12),
        ("oper", 18): d(2026, 11, 5, 18), ("enfo", 18): d(2026, 11, 5, 18),
    }
    assert freshest_complete_run(_fake_latest(avail), 240) == d(2026, 11, 6, 6)


def test_freshest_run_needs_both_hres_and_ens():
    from datetime import datetime as d
    from rainwin.fetch import freshest_complete_run
    avail = {
        ("oper", 0): d(2026, 11, 6, 0), ("enfo", 0): d(2026, 11, 6, 0),
        # today's 06 HRES is done but its ENS is still being published
        ("oper", 6): d(2026, 11, 6, 6), ("enfo", 6): d(2026, 11, 5, 6),
    }
    assert freshest_complete_run(_fake_latest(avail), 240) == d(2026, 11, 6, 0)
    with pytest.raises(RuntimeError):
        freshest_complete_run(_fake_latest({}), 240)
