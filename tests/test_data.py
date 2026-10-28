import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import pytest
from src.data_pipeline import (generate_data, load_config, check_data,
                               validate_input, FEATURES)

R = load_config(os.path.join(ROOT, "config/model.json"))["ranges"]


def test_reproducible():
    a = generate_data(500, seed=1, ranges=R)
    b = generate_data(500, seed=1, ranges=R)
    assert a.equals(b)


def test_different_seed():
    a = generate_data(500, seed=1, ranges=R)
    b = generate_data(500, seed=2, ranges=R)
    assert not a.equals(b)


def test_size_and_ranges():
    df = generate_data(1800, seed=42, ranges=R)
    assert len(df) >= 1500
    rep = check_data(df, R)
    assert sum(rep["missing"].values()) == 0
    assert sum(rep["out_of_range"].values()) == 0
    assert str(df["overdue_orders"].dtype).startswith("int")


def test_validate_bounds():
    low = {k: R[k][0] for k in FEATURES}
    high = {k: R[k][1] for k in FEATURES}
    assert validate_input(low, R)["utilization"] == 40
    assert validate_input(high, R)["utilization"] == 120


@pytest.mark.parametrize("bad", ["abc", None, float("nan"), float("inf"), 999, -5])
def test_validate_bad(bad):
    v = {k: (R[k][0] + R[k][1]) / 2 for k in FEATURES}
    v["utilization"] = bad
    with pytest.raises(ValueError):
        validate_input(v, R)
