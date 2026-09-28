"""Frozen N1a decision (docs/n1a_action_ambiguity_spec.md sec. 5-6) as a pure function. Inputs in percentage points.

per_base[b] = {"h": x, "MA": x, "MA_ci": (lo, hi), "MA_by_family": {f: x}, "MB": x, "Q": x, "Q_ci": (lo, hi),
               "support": {f: (n_repair, n_harm)}, "converged": bool}
"""
H_MIN, MA_MIN, MA_LO, MB_MIN, Q_MIN, SUPPORT = 1.0, 1.0, 0.5, 1.0, 0.5, 300
NAMES = {"STOP_HET": "STOP — insufficient action heterogeneity", "STOP_Z": "STOP — full Z already resolves the useful ambiguity",
         "GO": "GO — meaningful residual action ambiguity remains after full Z", "INC_VALID": "INCONCLUSIVE — validity",
         "INC_PREC": "INCONCLUSIVE — insufficient precision"}


def support_failures(q):
    return [f for f, (r, h) in q["support"].items() if r < SUPPORT or h < SUPPORT]


def go_base(q):
    return (q["MA"] >= MA_MIN and q["MA_ci"][0] >= MA_LO and sum(v >= MA_MIN for v in q["MA_by_family"].values()) >= 3
            and q["MB"] >= MB_MIN and q["Q"] >= Q_MIN and q["Q_ci"][0] > 0)


def stopz_base(q):
    return q["MA_ci"][1] < MA_MIN or q["Q_ci"][1] < Q_MIN


def decide(per_base: dict) -> dict:
    reasons = []
    for b, q in per_base.items():
        if not q["converged"]:
            reasons.append(f"base {b}: unconverged selector fit")
        if len(support_failures(q)) >= 2:
            reasons.append(f"base {b}: support < {SUPPORT} in families {support_failures(q)}")
    if reasons:
        return {"outcome": "INC_VALID", "name": NAMES["INC_VALID"], "reasons": reasons}
    if any(q["h"] < H_MIN for q in per_base.values()):
        return {"outcome": "STOP_HET", "name": NAMES["STOP_HET"], "reasons": [f"base {b}: h = {q['h']:.2f} pp" for b, q in per_base.items()]}
    if all(stopz_base(q) for q in per_base.values()):
        return {"outcome": "STOP_Z", "name": NAMES["STOP_Z"], "reasons": []}
    if all(go_base(q) for q in per_base.values()):
        return {"outcome": "GO", "name": NAMES["GO"], "reasons": []}
    return {"outcome": "INC_PREC", "name": NAMES["INC_PREC"], "reasons": [f"base {b}: go={go_base(q)} stopz={stopz_base(q)}" for b, q in per_base.items()]}
