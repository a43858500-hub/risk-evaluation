import json
import numpy as np
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .data_pipeline import FEATURES
from .fuzzy_systems import MamdaniSystem

# (переменная, терм, индекс параметра, нижняя граница, верхняя граница)
PARAMS = [
    ("utilization", "normal", 1, 70, 90),
    ("defect_rate", "medium", 1, 3.5, 6.5),
    ("downtime_pct", "medium", 1, 9, 15),
    ("overdue_orders", "some", 1, 11, 19),
    ("staff_availability", "medium", 1, 78, 88),
    ("energy_overrun", "normal", 1, 1, 7),
    ("risk_score", "moderate", 1, 30, 50),
    ("risk_score", "high", 1, 58, 78),
]


def metrics(y, pred):
    return {"MAE": mean_absolute_error(y, pred), "RMSE": mean_squared_error(y, pred) ** 0.5,
            "R2": r2_score(y, pred)}


def build_mf(base_mf, x):
    return base_mf.with_updates([(v, t, i, val) for (v, t, i, lo, hi), val in zip(PARAMS, x)])


def split_data(df, seed=42):
    return train_test_split(df, test_size=0.3, random_state=seed)


def optimize(config, df, seed=42, maxiter=25):
    train, test = split_data(df, seed)
    base = MamdaniSystem(config)

    def loss(x):
        mf = build_mf(base.mf, x)
        if mf.check():
            return 1e6
        sys_ = MamdaniSystem(config, mf=mf)
        pred = sys_.predict_many(train)
        return float(np.sqrt(np.mean((pred - train["expert_risk"].values) ** 2)))

    x0 = [base.mf.term(v, t)["params"][i] for v, t, i, lo, hi in PARAMS]
    bounds = [(lo, hi) for v, t, i, lo, hi in PARAMS]
    res = differential_evolution(loss, bounds, seed=seed, maxiter=maxiter, popsize=10,
                                 tol=1e-4, polish=False, x0=x0)
    tuned = MamdaniSystem(config, mf=build_mf(base.mf, res.x))
    out = {"x0": x0, "x": list(res.x), "train_loss_before": loss(x0), "train_loss_after": float(res.fun)}
    for name, part in (("train", train), ("test", test)):
        out[name + "_before"] = metrics(part["expert_risk"], base.predict_many(part))
        out[name + "_after"] = metrics(part["expert_risk"], tuned.predict_many(part))
    out["tuned_system"] = tuned
    out["test_df"] = test
    return out


def save_tuned(config, tuned_system, path):
    cfg = tuned_system.mf.to_config(config)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
