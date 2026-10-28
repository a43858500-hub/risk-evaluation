from abc import ABC, abstractmethod
from functools import reduce
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from .data_pipeline import FEATURES, validate_input
from .membership import MembershipFunctions, OUT
from .rules import RULES, OUT_RU


class FuzzySystem(ABC):
    def __init__(self, config, mf=None, rules=None):
        self.config = config
        self.mf = mf or MembershipFunctions(config)
        self.rules = rules or RULES

    @abstractmethod
    def predict(self, inputs):
        pass

    @abstractmethod
    def predict_many(self, data):
        pass

    def classify(self, score):
        for lo, hi, name in self.config["categories"]:
            if lo <= score < hi:
                return name
        return self.config["categories"][-1][2]

    def firing(self, inputs):
        arr = {k: np.atleast_1d(np.asarray(inputs[k], dtype=float)) for k in FEATURES}
        return np.array([r.strength(self.mf, arr) for r in self.rules])

    def explain(self, inputs, top=3):
        clean = validate_input(inputs, self.config["ranges"])
        f = self.firing(clean)[:, 0]
        order = np.argsort(-f)[:top]
        lines = []
        for i in order:
            if f[i] <= 0:
                continue
            lines.append(f"{self.rules[i].id} (степень {f[i]:.2f}): {self.rules[i].text()}")
        return lines


class MamdaniSystem(FuzzySystem):
    def __init__(self, config, mf=None, rules=None):
        super().__init__(config, mf, rules)
        self._sim = None

    def _build(self):
        ants = {}
        for var in FEATURES:
            u = self.mf.universe(var)
            a = ctrl.Antecedent(u, var)
            for t in self.mf.terms(var):
                a[t] = self.mf.degree(var, t, u)
            ants[var] = a
        uo = self.mf.universe(OUT)
        cons = ctrl.Consequent(uo, OUT, defuzzify_method="centroid")
        for t in self.mf.terms(OUT):
            cons[t] = self.mf.degree(OUT, t, uo)

        def expr(node):
            if node[0] == "is":
                return ants[node[1]][node[2]]
            parts = [expr(n) for n in node[1]]
            if node[0] == "and":
                return reduce(lambda a, b: a & b, parts)
            return reduce(lambda a, b: a | b, parts)

        sk_rules = [ctrl.Rule(expr(r.ant), cons[r.out], label=r.id) for r in self.rules]
        self._sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(sk_rules))

    def predict(self, inputs):
        clean = validate_input(inputs, self.config["ranges"])
        if self._sim is None:
            self._build()
        for k, v in clean.items():
            self._sim.input[k] = v
        self._sim.compute()
        return float(self._sim.output[OUT])

    def predict_many(self, data):
        inputs = {k: np.asarray(data[k], dtype=float) for k in FEATURES}
        f = np.array([r.strength(self.mf, inputs) for r in self.rules])
        y = self.mf.universe(OUT)
        agg = np.zeros((f.shape[1], len(y)))
        for t in self.mf.terms(OUT):
            idx = [i for i, r in enumerate(self.rules) if r.out == t]
            if not idx:
                continue
            w = f[idx].max(axis=0)
            agg = np.maximum(agg, np.minimum(w[:, None], self.mf.degree(OUT, t, y)[None, :]))
        s = agg.sum(axis=1)
        res = (agg * y).sum(axis=1) / np.where(s == 0, 1, s)
        res[s == 0] = 50.0
        return res


class SugenoSystem(FuzzySystem):
    def __init__(self, config, mf=None, rules=None):
        super().__init__(config, mf, rules)
        self.consts = np.array([config["sugeno_consts"][r.out] for r in self.rules], dtype=float)

    def predict_many(self, data):
        inputs = {k: np.asarray(data[k], dtype=float) for k in FEATURES}
        f = np.array([r.strength(self.mf, inputs) for r in self.rules])
        s = f.sum(axis=0)
        res = (f * self.consts[:, None]).sum(axis=0) / np.where(s == 0, 1, s)
        res[s == 0] = 50.0
        return res

    def predict(self, inputs):
        clean = validate_input(inputs, self.config["ranges"])
        return float(self.predict_many({k: [v] for k, v in clean.items()})[0])


def compare_systems(mamdani, sugeno, inputs):
    m = mamdani.predict(inputs)
    s = sugeno.predict(inputs)
    return {"mamdani": m, "sugeno": s, "diff": abs(m - s),
            "category": mamdani.classify(m), "category_sugeno": sugeno.classify(s)}
