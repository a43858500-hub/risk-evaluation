import json
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEATURES = ["utilization", "defect_rate", "downtime_pct",
            "overdue_orders", "staff_availability", "energy_overrun"]


def load_config(path=None):
    path = path or os.path.join(ROOT, "config", "model.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def generate_data(n=1800, seed=42, ranges=None):
    rng = np.random.default_rng(seed)
    if ranges is None:
        ranges = load_config()["ranges"]

    mode = rng.choice(3, size=n, p=[0.45, 0.35, 0.20])
    util = np.where(mode == 0, rng.normal(80, 8, n),
           np.where(mode == 1, rng.normal(102, 8, n), rng.normal(65, 12, n)))
    defect = np.where(mode == 0, rng.gamma(2.0, 1.0, n),
             np.where(mode == 1, rng.gamma(3.0, 1.3, n), rng.gamma(4.0, 1.6, n)))
    down = np.where(mode == 0, rng.gamma(2.0, 2.0, n),
           np.where(mode == 1, rng.gamma(3.0, 2.5, n), rng.gamma(5.0, 3.0, n)))
    overdue = np.where(mode == 0, rng.poisson(4, n),
              np.where(mode == 1, rng.poisson(14, n), rng.poisson(22, n)))
    staff = np.where(mode == 0, rng.normal(92, 4, n),
            np.where(mode == 1, rng.normal(84, 6, n), rng.normal(76, 7, n)))
    energy = np.where(mode == 0, rng.normal(2, 3, n),
             np.where(mode == 1, rng.normal(9, 5, n), rng.normal(14, 6, n)))

    df = pd.DataFrame({
        "period": rng.integers(1, 13, n),
        "workshop": rng.integers(1, 9, n),
        "utilization": util,
        "defect_rate": defect,
        "downtime_pct": down,
        "overdue_orders": overdue,
        "staff_availability": staff,
        "energy_overrun": energy,
    })

    for col in FEATURES:
        lo, hi = ranges[col]
        df[col] = df[col].clip(lo, hi)
    df["overdue_orders"] = df["overdue_orders"].round().astype(int)

    u = df["utilization"]
    risk = (
        0.20 * np.abs(u - 80) / 40 * 100
        + 0.20 * df["defect_rate"] / 15 * 100
        + 0.20 * df["downtime_pct"] / 30 * 100
        + 0.20 * df["overdue_orders"] / 40 * 100
        + 0.12 * (100 - df["staff_availability"]) / 40 * 100
        + 0.08 * (df["energy_overrun"] + 10) / 40 * 100
    )
    risk = risk * 1.35 + rng.normal(0, 4, n)
    df["expert_risk"] = np.clip(risk, 0, 100)

    for col in ["utilization", "defect_rate", "downtime_pct",
                "staff_availability", "energy_overrun", "expert_risk"]:
        df[col] = df[col].round(2)
    return df


def save_data(df, path=None):
    path = path or os.path.join(ROOT, "data", "production.csv")
    df.to_csv(path, index=False)


def load_data(path=None):
    path = path or os.path.join(ROOT, "data", "production.csv")
    return pd.read_csv(path)


def check_data(df, ranges):
    report = {"shape": df.shape, "dtypes": df.dtypes.astype(str).to_dict(),
              "missing": df.isna().sum().to_dict(), "out_of_range": {}, "outliers_iqr": {}}
    for col, (lo, hi) in ranges.items():
        report["out_of_range"][col] = int(((df[col] < lo) | (df[col] > hi)).sum())
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        mask = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
        report["outliers_iqr"][col] = int(mask.sum())
    return report


def describe_data(df):
    cols = FEATURES + ["expert_risk"]
    return df[cols].describe().T, df[cols].corr()


def validate_input(values, ranges):
    clean = {}
    for name in FEATURES:
        if name not in values:
            raise ValueError(f"нет значения: {name}")
        try:
            x = float(values[name])
        except (TypeError, ValueError):
            raise ValueError(f"{name}: нужно число")
        if np.isnan(x) or np.isinf(x):
            raise ValueError(f"{name}: недопустимое значение")
        lo, hi = ranges[name]
        if x < lo or x > hi:
            raise ValueError(f"{name}: значение должно быть от {lo} до {hi}")
        clean[name] = x
    return clean
