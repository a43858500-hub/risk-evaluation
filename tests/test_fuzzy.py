import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import pytest
from src.data_pipeline import load_config, FEATURES, generate_data
from src.fuzzy_systems import MamdaniSystem, SugenoSystem, FuzzySystem, compare_systems
from src.membership import MembershipFunctions
from src.rules import RULES

cfg = load_config(os.path.join(ROOT, "config/model.json"))
R = cfg["ranges"]
mamdani = MamdaniSystem(cfg)
sugeno = SugenoSystem(cfg)

NORMAL = dict(utilization=80, defect_rate=1, downtime_pct=3, overdue_orders=2, staff_availability=95, energy_overrun=2)
STRAIN = dict(utilization=100, defect_rate=6, downtime_pct=14, overdue_orders=15, staff_availability=82, energy_overrun=9)
CRIT = dict(utilization=112, defect_rate=12, downtime_pct=25, overdue_orders=32, staff_availability=65, energy_overrun=22)


def test_structure():
    assert len(RULES) >= 18
    assert [r.id for r in RULES][:3] == ["R01", "R02", "R03"]
    assert isinstance(mamdani, FuzzySystem) and isinstance(sugeno, FuzzySystem)
    assert len(MembershipFunctions(cfg).terms("risk_score")) >= 4


def test_mf_params_valid():
    assert MembershipFunctions(cfg).check() == []


def test_mf_crossing_defect():
    mf = MembershipFunctions(cfg)
    x = 2.6
    a = mf.degree("defect_rate", "low", x)
    b = mf.degree("defect_rate", "medium", x)
    assert a == pytest.approx(0.2) and b == pytest.approx(0.2)


@pytest.mark.parametrize("var", FEATURES)
def test_mf_neighbors_cross(var):
    mf = MembershipFunctions(cfg)
    terms = mf.terms(var)
    u = mf.universe(var)
    for t1, t2 in zip(terms[:-1], terms[1:]):
        a, b = mf.degree(var, t1, u), mf.degree(var, t2, u)
        i = np.argmin(np.abs(a - b) + (np.minimum(a, b) == 0) * 10)
        assert 0 < a[i] < 1 and abs(a[i] - b[i]) < 0.02


def test_scenarios_order():
    n, s, c = (mamdani.predict(x) for x in (NORMAL, STRAIN, CRIT))
    assert n < s < c
    assert c - n > 40
    assert mamdani.classify(n) == "низкий"
    assert mamdani.classify(c) == "критический"


def test_extreme_inputs_no_nan():
    lo = {k: R[k][0] for k in FEATURES}
    hi = {k: R[k][1] for k in FEATURES}
    for x in (lo, hi):
        for sys_ in (mamdani, sugeno):
            v = sys_.predict(x)
            assert np.isfinite(v) and 0 <= v <= 100


def test_random_grid_all_rules_cover():
    df = generate_data(300, seed=7, ranges=R)
    for sys_ in (mamdani, sugeno):
        v = sys_.predict_many(df)
        assert np.isfinite(v).all()
        assert (v != 50.0).all()


def test_invalid_inputs():
    bad = dict(NORMAL, utilization=500)
    with pytest.raises(ValueError):
        mamdani.predict(bad)
    with pytest.raises(ValueError):
        sugeno.predict(dict(NORMAL, defect_rate="abc"))
    with pytest.raises(ValueError):
        mamdani.predict(dict(NORMAL, downtime_pct=float("nan")))


def test_sugeno_reference_values():
    assert sugeno.predict(dict(utilization=80, defect_rate=0, downtime_pct=0, overdue_orders=0,
                               staff_availability=100, energy_overrun=4)) == pytest.approx(10.0)
    x = dict(utilization=80, defect_rate=5, downtime_pct=21, overdue_orders=15,
             staff_availability=83, energy_overrun=4)
    assert sugeno.predict(x) == pytest.approx(47.0)


def test_vectorized_matches_skfuzzy():
    df = generate_data(12, seed=3, ranges=R)
    fast = mamdani.predict_many(df)
    slow = np.array([mamdani.predict(df.iloc[i][FEATURES].to_dict()) for i in range(len(df))])
    assert np.abs(fast - slow).max() < 0.5


def test_compare_and_explain():
    res = compare_systems(mamdani, sugeno, STRAIN)
    assert res["diff"] == pytest.approx(abs(res["mamdani"] - res["sugeno"]))
    assert len(mamdani.explain(STRAIN)) > 0
