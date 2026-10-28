import os
import sys
import json
import numpy as np
import pandas as pd

from src.data_pipeline import (ROOT, FEATURES, load_config, generate_data, save_data, load_data,
                               check_data, describe_data)
from src.fuzzy_systems import MamdaniSystem, SugenoSystem, compare_systems
from src.rules import RULES, T_NORM, T_CONORM
from src import visualization as viz
from src import clustering, optimization

OUT = os.path.join(ROOT, "output")

SCENARIOS = {
    "нормальный": dict(utilization=80, defect_rate=1, downtime_pct=3, overdue_orders=2,
                       staff_availability=95, energy_overrun=2),
    "напряжённый": dict(utilization=100, defect_rate=6, downtime_pct=14, overdue_orders=15,
                        staff_availability=82, energy_overrun=9),
    "критический": dict(utilization=112, defect_rate=12, downtime_pct=25, overdue_orders=32,
                        staff_availability=65, energy_overrun=22),
}


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = load_config()
    path = os.path.join(ROOT, cfg["data_path"])
    if not os.path.exists(path):
        save_data(generate_data(cfg["n_samples"], cfg["seed"], cfg["ranges"]), path)
    df = load_data(path)

    print("--- 1. Анализ данных ---")
    rep = check_data(df, cfg["ranges"])
    print("размер:", rep["shape"], "| пропуски:", sum(rep["missing"].values()),
          "| вне диапазона:", sum(rep["out_of_range"].values()))
    print("выбросы (IQR):", rep["outliers_iqr"])
    desc, corr = describe_data(df)
    desc.round(2).to_csv(os.path.join(OUT, "describe.csv"))
    corr.round(3).to_csv(os.path.join(OUT, "correlations.csv"))
    print(desc.round(2))
    viz.eda_plots(df, OUT)

    print("\n--- 2. Нечёткая модель ---")
    print(f"t-норма (AND) = {T_NORM}, t-конорма (OR) = {T_CONORM}, правил: {len(RULES)}")
    m_sys, s_sys = MamdaniSystem(cfg), SugenoSystem(cfg)
    viz.mf_plots(m_sys.mf, OUT)
    viz.rules_graph(RULES, OUT)
    scen = {}
    for name, x in SCENARIOS.items():
        r = compare_systems(m_sys, s_sys, x)
        scen[name] = r
        print(f"{name}: Мамдани={r['mamdani']:.1f}, Сугено={r['sugeno']:.1f}, "
              f"|diff|={r['diff']:.1f}, категория: {r['category']}")
    mv, sv = m_sys.predict_many(df), s_sys.predict_many(df)
    viz.compare_plot(mv, sv, OUT, scen)
    print(f"среднее расхождение Мамдани/Сугено по данным: {np.mean(np.abs(mv - sv)):.2f}")

    print("\n--- 3. Нечёткая кластеризация (FCM) ---")
    table, models, scaler, best = clustering.run_fcm(df)
    print(table.round(4).to_string(index=False))
    print("выбрано кластеров:", best)
    cntr, u = models[best]
    centers = clustering.interpret_centers(cntr, scaler, m_sys)
    print(centers.to_string(index=False))
    mt = clustering.membership_table(u)
    print(mt)
    table.to_csv(os.path.join(OUT, "fcm_metrics.csv"), index=False)
    centers.to_csv(os.path.join(OUT, "fcm_centers.csv"), index=False)
    mt.to_csv(os.path.join(OUT, "fcm_memberships.csv"))
    viz.cluster_plots(table, cntr, scaler, OUT)

    print("\n--- 4. Оптимизация параметров ---")
    res = optimization.optimize(cfg, df)
    for part in ("train", "test"):
        print(part, "до:", {k: round(v, 3) for k, v in res[part + "_before"].items()})
        print(part, "после:", {k: round(v, 3) for k, v in res[part + "_after"].items()})
    for (v, t, i, lo, hi), a, b in zip(optimization.PARAMS, res["x0"], res["x"]):
        print(f"{v}.{t}[{i}]: {a:.2f} -> {b:.2f}  (границы {lo}..{hi})")
    test = res["test_df"]
    viz.optimization_plot(test["expert_risk"], m_sys.predict_many(test), res["tuned_system"].predict_many(test), OUT)
    viz.mf_plots(res["tuned_system"].mf, OUT, "membership_tuned.png")
    optimization.save_tuned(cfg, res["tuned_system"], os.path.join(ROOT, "config", "model_tuned.json"))
    summary = {k: res[k] for k in ("train_before", "train_after", "test_before", "test_after", "x0", "x")}
    with open(os.path.join(OUT, "optimization.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\nГотово, результаты в папке output/. Интерфейс: python main.py gui")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "gui":
        from src.gui import run
        run()
    else:
        main()
