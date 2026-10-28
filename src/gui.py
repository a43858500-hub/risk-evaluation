import os
import json
import tkinter as tk
from tkinter import ttk, messagebox

from .data_pipeline import FEATURES, ROOT, load_config, validate_input
from .fuzzy_systems import MamdaniSystem, SugenoSystem
from .rules import RULES

LABELS = {"utilization": "Загрузка мощности, %", "defect_rate": "Доля брака, %",
          "downtime_pct": "Простои оборудования, %", "overdue_orders": "Просроченные заказы, шт",
          "staff_availability": "Доступность персонала, %", "energy_overrun": "Перерасход энергии, %"}

SCENARIOS = {
    "Нормальный": dict(utilization=80, defect_rate=1, downtime_pct=3, overdue_orders=2,
                       staff_availability=95, energy_overrun=2),
    "Напряжённый": dict(utilization=100, defect_rate=6, downtime_pct=14, overdue_orders=15,
                        staff_availability=82, energy_overrun=9),
    "Критический": dict(utilization=112, defect_rate=12, downtime_pct=25, overdue_orders=32,
                        staff_availability=65, energy_overrun=22),
}


class App:
    def __init__(self, root):
        self.root = root
        root.title("Оценка производственного риска цеха")
        self.base_cfg = load_config()
        self.tuned_path = os.path.join(ROOT, "config", "model_tuned.json")
        self.use_tuned = tk.BooleanVar(value=False)
        self.systems = {}
        self.entries = {}

        frm = ttk.Frame(root, padding=10)
        frm.grid(row=0, column=0, sticky="nsew")
        ranges = self.base_cfg["ranges"]
        for i, name in enumerate(FEATURES):
            lo, hi = ranges[name]
            ttk.Label(frm, text=f"{LABELS[name]} [{lo}..{hi}]").grid(row=i, column=0, sticky="w", pady=2)
            e = ttk.Entry(frm, width=12)
            e.grid(row=i, column=1, padx=8)
            self.entries[name] = e

        row = len(FEATURES)
        bf = ttk.Frame(frm)
        bf.grid(row=row, column=0, columnspan=2, pady=6, sticky="w")
        for n in SCENARIOS:
            ttk.Button(bf, text=n, command=lambda k=n: self.fill(k)).pack(side="left", padx=2)
        if os.path.exists(self.tuned_path):
            ttk.Checkbutton(frm, text="Настроенные параметры (после оптимизации)",
                            variable=self.use_tuned).grid(row=row + 1, column=0, columnspan=2, sticky="w")
        ttk.Button(frm, text="Рассчитать", command=self.calc).grid(row=row + 2, column=0, columnspan=2, pady=6)

        self.res = {}
        for i, key in enumerate(["Мамдани", "Сугено", "Расхождение"]):
            ttk.Label(frm, text=key + ":").grid(row=row + 3 + i, column=0, sticky="w")
            lab = ttk.Label(frm, text="-", font=("Arial", 11, "bold"))
            lab.grid(row=row + 3 + i, column=1, sticky="w")
            self.res[key] = lab
        self.text = tk.Text(frm, width=70, height=9, wrap="word")
        self.text.grid(row=row + 7, column=0, columnspan=2, pady=6)
        self.fill("Нормальный")

    def fill(self, name):
        for k, v in SCENARIOS[name].items():
            self.entries[k].delete(0, tk.END)
            self.entries[k].insert(0, str(v))

    def get_systems(self):
        tuned = self.use_tuned.get() and os.path.exists(self.tuned_path)
        if tuned not in self.systems:
            if tuned:
                with open(self.tuned_path, encoding="utf-8") as f:
                    cfg = json.load(f)
            else:
                cfg = self.base_cfg
            self.systems[tuned] = (MamdaniSystem(cfg), SugenoSystem(cfg))
        return self.systems[tuned]

    def calc(self):
        raw = {k: e.get().replace(",", ".") for k, e in self.entries.items()}
        try:
            clean = validate_input(raw, self.base_cfg["ranges"])
            m_sys, s_sys = self.get_systems()
            m = m_sys.predict(clean)
            s = s_sys.predict(clean)
        except ValueError as err:
            messagebox.showerror("Ошибка ввода", str(err))
            return
        except Exception as err:
            messagebox.showerror("Ошибка расчёта", str(err))
            return
        self.res["Мамдани"].config(text=f"{m:.1f} ({m_sys.classify(m)})")
        self.res["Сугено"].config(text=f"{s:.1f} ({s_sys.classify(s)})")
        self.res["Расхождение"].config(text=f"{abs(m - s):.1f}")
        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, "Наиболее активные правила:\n" + "\n".join(m_sys.explain(clean)))


def run():
    root = tk.Tk()
    App(root)
    root.mainloop()
