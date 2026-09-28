"""Frozen N1a-DP decision (docs/n1a_dp_spec.md sec. 6-7) as a pure function. Inputs in pp (phi as a fraction).

per[family][base] = {"phi": x, "phi_ci": (lo, hi), "U": x, "U_ci": (lo, hi), "MA": x, "MA_ci": (lo, hi)}
MB[base] = family-independent local-Z ambiguity (pp); valid = set of valid families.
"""
PHI_RES, MA_RES = 0.5, 3.5            # RESOLVED: phi lower >= 0.5 and M_A upper <= 3.5 pp
U_SUB, MA_SUB = 4.0, 2.0              # SUBSTANTIAL: unrecovered opportunity lower >= 4.0 pp and M_A lower >= 2.0 pp
MB_MIN = 1.0
MIN_VALID = 5
NAMES = {"LIMIT": "OUTPUT-DECODER LIMITATION", "ROBUST": "PANEL-ROBUST OUTPUT AMBIGUITY", "SPECIFIC": "FAMILY-SPECIFIC",
         "INC_VALID": "INCONCLUSIVE (validity)", "INC": "INCONCLUSIVE"}


def resolved(q):
    return q["phi_ci"][0] >= PHI_RES and q["MA_ci"][1] <= MA_RES


def substantial(q):
    return q["U_ci"][0] >= U_SUB and q["MA_ci"][0] >= MA_SUB


def decide(per, MB, valid, bases=(2, 4)):
    V = sorted(valid)
    lab = {f: {b: ("RESOLVED" if resolved(per[f][b]) else "SUBSTANTIAL" if substantial(per[f][b]) else "INTERMEDIATE") for b in bases} for f in per}
    out = {"labels": lab, "valid_families": V}
    if len(V) < MIN_VALID:
        return dict(out, outcome="INC_VALID", name=NAMES["INC_VALID"])
    res_both = [f for f in V if all(lab[f][b] == "RESOLVED" for b in bases)]
    res_any = [f for f in V if any(lab[f][b] == "RESOLVED" for b in bases)]
    sub_all = all(lab[f][b] == "SUBSTANTIAL" for f in V for b in bases)
    out.update(resolved_both=res_both, resolved_any=res_any)
    if len(res_both) >= 2:
        return dict(out, outcome="LIMIT", name=NAMES["LIMIT"])
    if sub_all and not res_any and all(MB[b] >= MB_MIN for b in bases):
        return dict(out, outcome="ROBUST", name=NAMES["ROBUST"])
    if res_any:
        return dict(out, outcome="SPECIFIC", name=NAMES["SPECIFIC"])
    return dict(out, outcome="INC", name=NAMES["INC"])
