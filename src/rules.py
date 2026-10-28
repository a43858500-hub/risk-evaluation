import numpy as np

T_NORM = "min"
T_CONORM = "max"

U, D, T, O, S, E = ("utilization", "defect_rate", "downtime_pct",
                    "overdue_orders", "staff_availability", "energy_overrun")

PHRASE = {
    (U, "low"): "загрузка низкая", (U, "normal"): "загрузка нормальная", (U, "high"): "загрузка высокая",
    (D, "low"): "брак низкий", (D, "medium"): "брак средний", (D, "high"): "брак высокий",
    (T, "low"): "простои малые", (T, "medium"): "простои средние", (T, "high"): "простои большие",
    (O, "few"): "просрочек мало", (O, "some"): "просрочек немного", (O, "many"): "просрочек много",
    (S, "low"): "персонала мало", (S, "medium"): "персонала достаточно", (S, "high"): "персонала в норме",
    (E, "saving"): "энергия в экономии", (E, "normal"): "энергия в норме", (E, "excess"): "энергия в перерасходе",
}
OUT_RU = {"low": "низкий", "moderate": "умеренный", "high": "высокий", "critical": "критический"}


def IS(v, t):
    return ("is", v, t)


def AND(*a):
    return ("and", list(a))


def OR(*a):
    return ("or", list(a))


def _eval(node, mf, inputs):
    if node[0] == "is":
        return mf.degree(node[1], node[2], inputs[node[1]])
    vals = [_eval(n, mf, inputs) for n in node[1]]
    return np.minimum.reduce(vals) if node[0] == "and" else np.maximum.reduce(vals)


def _text(node):
    if node[0] == "is":
        return PHRASE[(node[1], node[2])]
    word = " И " if node[0] == "and" else " ИЛИ "
    s = word.join(_text(n) for n in node[1])
    return "(" + s + ")" if len(node[1]) > 1 else s


class Rule:
    def __init__(self, rid, ant, out):
        self.id = rid
        self.ant = ant
        self.out = out

    def strength(self, mf, inputs):
        return _eval(self.ant, mf, inputs)

    def text(self):
        t = _text(self.ant)
        if t.startswith("(") and t.endswith(")"):
            t = t[1:-1]
        return f"ЕСЛИ {t}, ТО риск {OUT_RU[self.out]}"

    def inputs_used(self):
        res = []

        def walk(n):
            if n[0] == "is":
                res.append((n[1], n[2]))
            else:
                for x in n[1]:
                    walk(x)
        walk(self.ant)
        return res


RULES_DEF = [
    ("R01", AND(IS(U, "normal"), IS(D, "low"), IS(T, "low")), "low"),
    ("R02", AND(IS(U, "normal"), IS(O, "few"), IS(S, "high")), "low"),
    ("R03", AND(IS(U, "normal"), IS(D, "low"), IS(E, "normal")), "low"),
    ("R04", AND(IS(U, "normal"), IS(D, "medium"), IS(T, "medium")), "moderate"),
    ("R05", AND(IS(U, "normal"), IS(O, "some")), "moderate"),
    ("R06", AND(IS(U, "high"), IS(S, "high"), IS(D, "low")), "moderate"),
    ("R07", AND(IS(U, "high"), IS(T, "medium")), "high"),
    ("R08", AND(IS(U, "high"), IS(O, "some"), IS(S, "medium")), "high"),
    ("R09", AND(IS(U, "high"), OR(IS(D, "high"), IS(T, "high"))), "critical"),
    ("R10", AND(IS(U, "high"), IS(O, "many")), "critical"),
    ("R11", AND(IS(U, "low"), IS(D, "low"), IS(T, "low")), "moderate"),
    ("R12", AND(IS(U, "low"), IS(S, "low")), "high"),
    ("R13", AND(IS(U, "low"), IS(T, "high")), "critical"),
    ("R14", AND(IS(D, "high"), IS(T, "high")), "critical"),
    ("R15", OR(IS(D, "high"), IS(T, "high")), "high"),
    ("R16", AND(IS(S, "low"), IS(O, "many")), "critical"),
    ("R17", AND(IS(S, "low"), IS(O, "some")), "high"),
    ("R18", AND(IS(E, "excess"), IS(U, "high")), "high"),
    ("R19", AND(IS(E, "excess"), IS(T, "medium")), "high"),
    ("R20", AND(IS(U, "low"), IS(E, "saving")), "moderate"),
    ("R21", AND(IS(D, "medium"), IS(T, "medium"), IS(S, "medium")), "moderate"),
    ("R22", AND(IS(D, "low"), IS(T, "low"), IS(O, "few"), IS(S, "high")), "low"),
    ("R23", AND(IS(O, "many"), IS(T, "high")), "critical"),
    ("R24", AND(IS(U, "normal"), IS(S, "low")), "high"),
    ("R25", OR(IS(D, "medium"), IS(T, "medium"), IS(O, "some"), IS(S, "medium")), "moderate"),
    ("R26", AND(IS(D, "low"), IS(T, "low"), IS(O, "few")), "low"),
    ("R27", IS(O, "many"), "high"),
]

RULES = [Rule(r, a, o) for r, a, o in RULES_DEF]
