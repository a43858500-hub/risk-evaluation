import numpy as np
import pandas as pd
import skfuzzy as fuzz
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from .data_pipeline import FEATURES

NAMES_RU = {"utilization": "загрузка", "defect_rate": "брак", "downtime_pct": "простои",
            "overdue_orders": "просрочки", "staff_availability": "персонал", "energy_overrun": "перерасход энергии"}


def run_fcm(df, c_values=range(2, 7), m=2.0, seed=42):
    scaler = StandardScaler().fit(df[FEATURES].values)
    X = scaler.transform(df[FEATURES].values)
    rows, models = [], {}
    for c in c_values:
        cntr, u, _, _, _, _, fpc = fuzz.cluster.cmeans(X.T, c, m, error=1e-5, maxiter=300, seed=seed)
        labels = u.argmax(axis=0)
        sil = silhouette_score(X, labels)
        rows.append({"c": c, "FPC": fpc, "Silhouette": sil})
        models[c] = (cntr, u)
    table = pd.DataFrame(rows)
    best_c = int(table.loc[table["Silhouette"].idxmax(), "c"])
    return table, models, scaler, best_c


def interpret_centers(cntr, scaler, system):
    centers = scaler.inverse_transform(cntr)
    res = []
    for i, row in enumerate(centers):
        values = dict(zip(FEATURES, row))
        score = float(system.predict_many({k: [v] for k, v in values.items()})[0])
        notes = []
        for j, f in enumerate(FEATURES):
            z = cntr[i, j]
            if z > 0.6:
                notes.append(f"{NAMES_RU[f]} выше среднего")
            elif z < -0.6:
                notes.append(f"{NAMES_RU[f]} ниже среднего")
        res.append({"cluster": i, **{f: round(values[f], 2) for f in FEATURES},
                    "risk_mamdani": round(score, 1), "category": system.classify(score),
                    "description": "; ".join(notes) if notes else "показатели около среднего"})
    return pd.DataFrame(res)


def membership_table(u, n=5):
    t = pd.DataFrame(u[:, :n].T, columns=[f"cluster_{i}" for i in range(u.shape[0])]).round(3)
    t.index.name = "obs"
    return t
