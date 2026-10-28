import copy
import numpy as np
import skfuzzy as fuzz

OUT = "risk_score"


class MembershipFunctions:
    def __init__(self, config):
        self.inputs = copy.deepcopy(config["inputs"])
        self.output = copy.deepcopy(config["output"]["terms"])
        self.ranges = config["ranges"]
        self.out_range = config["output"]["range"]
        self._cache = {}

    def terms(self, var):
        return list(self.output if var == OUT else self.inputs[var])

    def term(self, var, name):
        return self.output[name] if var == OUT else self.inputs[var][name]

    def universe(self, var):
        if var == OUT:
            return np.arange(self.out_range[0], self.out_range[1] + 0.5, 0.5)
        lo, hi = self.ranges[var]
        return np.linspace(lo, hi, 1001)

    def degree(self, var, name, x):
        t = self.term(var, name)
        arr = np.atleast_1d(np.asarray(x, dtype=float))
        if t["type"] == "trimf":
            y = fuzz.trimf(arr, t["params"])
        else:
            y = fuzz.trapmf(arr, t["params"])
        return y.reshape(np.shape(x)) if np.ndim(x) else float(y[0])

    def peak(self, var, name):
        p = self.term(var, name)["params"]
        return p[1] if len(p) == 3 else (p[1] + p[2]) / 2

    def with_updates(self, updates):
        new = MembershipFunctions({"inputs": self.inputs, "output": {"terms": self.output, "range": self.out_range},
                                   "ranges": self.ranges})
        for var, name, idx, value in updates:
            new.term(var, name)["params"][idx] = float(value)
        return new

    def check(self):
        errors = []
        for var in list(self.inputs) + [OUT]:
            lo, hi = self.out_range if var == OUT else self.ranges[var]
            peaks = []
            for name in self.terms(var):
                p = self.term(var, name)["params"]
                if any(p[i] > p[i + 1] for i in range(len(p) - 1)):
                    errors.append(f"{var}.{name}: параметры не по возрастанию")
                if p[0] < lo or p[-1] > hi:
                    errors.append(f"{var}.{name}: выход за носитель")
                peaks.append(self.peak(var, name))
            if any(peaks[i] >= peaks[i + 1] for i in range(len(peaks) - 1)):
                errors.append(f"{var}: нарушен порядок термов")
        return errors

    def to_config(self, config):
        cfg = copy.deepcopy(config)
        cfg["inputs"] = copy.deepcopy(self.inputs)
        cfg["output"]["terms"] = copy.deepcopy(self.output)
        return cfg
