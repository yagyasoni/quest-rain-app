import numpy as np

from rainwin.insights import dry_spells, plan_check


def test_dry_spells_merge_within_tolerance_and_never_overlap():
    starts = np.arange(10.0)
    #            0   1  2   3   4   5   6   7  8  9
    curve = np.array([0, 3, 0, 50, 60, 50, 10, 8, 9, 30], float)
    spells = dry_spells(curve, starts, np.ones(10, bool), duration_h=2)
    # 0,3,0 are within 5 points of 0 -> one spell from 0 to 2+2=4 h
    assert spells[0] == {"start_h": 0.0, "end_h": 4.0, "chance": 3.0}
    # next driest free start is 7 (8%): grows over 6 (10) and 8 (9), not 9 (30)
    assert spells[1] == {"start_h": 6.0, "end_h": 10.0, "chance": 10.0}
    for a in spells:
        for b in spells:
            if a is not b:
                assert a["end_h"] <= b["start_h"] or b["end_h"] <= a["start_h"]


def test_dry_spells_do_not_bridge_gaps_in_allowed_hours():
    starts = np.array([20.0, 21.0, 30.0, 31.0])   # evening, then (after the night) next morning
    curve = np.zeros(4)
    spells = dry_spells(curve, starts, np.ones(4, bool), duration_h=1)
    assert [(s["start_h"], s["end_h"]) for s in spells] == [(20.0, 22.0), (30.0, 32.0)]


def test_dry_spells_honours_allowed_mask():
    starts = np.arange(4.0)
    spells = dry_spells(np.zeros(4), starts, np.array([False, True, True, False]), duration_h=1)
    assert spells == [{"start_h": 1.0, "end_h": 3.0, "chance": 0.0}]


def test_plan_check_area_chance_is_at_least_point_chance():
    steps = np.array([0.0, 3.0])
    point = np.array([[1.0], [0.0], [0.0], [0.0]])            # 4 members, 1 interval
    area = np.zeros((4, 1, 3, 3))
    area[:, :, 1, 1] = point                                  # centre cell = the point
    area[1, 0, 0, 0] = 1.0                                    # member 2 rains in a corner cell
    hres = np.zeros(1)                                        # one interval
    r = plan_check(hres, point, steps, 0, 3, 1.0, area_members=area)
    assert r["chance"] == 25 and r["area_chance"] == 50
