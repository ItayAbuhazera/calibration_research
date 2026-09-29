"""N1a-DP decision (docs/n1a_dp_spec.md sec. 6-7, DRAFT r4 — NOT FROZEN, NOT AUTHORIZED) as pure functions. Inputs in pp (phi as a fraction).

per[family][base] = {"phi_ci": (lo, hi), "U_ci": (lo, hi), "MA_ci": (lo, hi), ...}
informative[family][base] = bool (I1/I2 rule, sec. 6.2); MB[base] = local-Z ambiguity (pp); valid = set of valid families.
NOISE[family] = {"phi": s, "U": s, "MA": s, "inner_margin_pp": s} for stochastic families (filled from the audit at freeze; sec. 6.4).
"""
PHI_RES, MA_RES = 0.5, 3.5
U_SUB, MA_SUB = 4.0, 2.0
MB_MIN = 1.0
FAMILIES5 = ("linear", "rff", "lgbm", "mlp", "knn")   # prospective five-family N1a-DP panel (poly2 removed prospectively, r4)
MIN_VALID = 4                    # at most one invalid family (r3: 5 of 6; same tolerance of one invalid family)
MIN_INFORMATIVE = 4
TAU_FLOOR = 0.10
I_UNITS, I_FOLDS = 16, 3            # informativeness: >= 16/20 units overall and >= 3/5 folds in every held-out family
ROUTE_LO, ROUTE_HI = 0.01, 0.99
GROUPS = {"linear": "l2head", "rff": "l2head", "lgbm": "trees", "mlp": "neural", "knn": "local"}
STOCHASTIC = ("lgbm", "mlp")
# sec. 6.4: s_m(q) = max per-unit full-HPO replicate SD over the 4 audit units (results/n1adp_audit/summary_5family.json)
NOISE = {"lgbm": {"inner_margin_pp": 0.06719696495203807, "phi": 0.0240058088639461, "U": 0.18804550276757762, "MA": 0.17183648558376055}, "mlp": {"inner_margin_pp": 0.061377139753354314, "phi": 0.04198018459459951, "U": 0.3288447793243627, "MA": 0.24041839957032365}}
NAMES = {"LIMIT": "OUTPUT-DECODER LIMITATION", "ROBUST": "PANEL-ROBUST OUTPUT AMBIGUITY", "SPECIFIC": "FAMILY-SPECIFIC",
         "INC_VALID": "INCONCLUSIVE (validity)", "INC": "INCONCLUSIVE"}


def noise(family, q, table=None):
    t = NOISE if table is None else table
    if family not in STOCHASTIC:
        return 0.0
    assert t is not None, "NOISE must be filled from the audit before N1a-DP is frozen"
    return float(t[family][q])


def tau(family, table=None):
    return max(TAU_FLOOR, 2 * noise(family, "inner_margin_pp", table))


def unit_passes(route_rate, inner_margin_pp, family, table=None):
    return (ROUTE_LO <= route_rate <= ROUTE_HI) and inner_margin_pp >= tau(family, table)


def informative(unit_pass_by_heldout):
    """unit_pass_by_heldout: {heldout family: [bool per fold]} (4 x 5). >= 16/20 overall and >= 3/5 within every held-out family."""
    total = sum(sum(v) for v in unit_pass_by_heldout.values())
    return total >= I_UNITS and all(sum(v) >= I_FOLDS for v in unit_pass_by_heldout.values())


def resolved(q, family, table=None):
    return q["phi_ci"][0] - 2 * noise(family, "phi", table) >= PHI_RES and q["MA_ci"][1] + 2 * noise(family, "MA", table) <= MA_RES


def substantial(q, family, table=None):
    return q["U_ci"][0] - 2 * noise(family, "U", table) >= U_SUB and q["MA_ci"][0] - 2 * noise(family, "MA", table) >= MA_SUB


def label(q, family, table=None):
    return "RESOLVED" if resolved(q, family, table) else "SUBSTANTIAL" if substantial(q, family, table) else "INTERMEDIATE"


def decide(per, MB, valid, info, validity_extra_ok=True, bases=(2, 4), table=None):
    V = sorted(valid)
    lab = {f: {b: label(per[f][b], f, table) for b in bases} for f in per}
    out = {"labels": lab, "valid_families": V, "informative": info}
    if len(V) < MIN_VALID or not validity_extra_ok:
        return dict(out, outcome="INC_VALID", name=NAMES["INC_VALID"])
    res_both = [f for f in V if all(lab[f][b] == "RESOLVED" for b in bases)]
    res_any = [f for f in V if any(lab[f][b] == "RESOLVED" for b in bases)]
    inf_both = [f for f in V if all(info[f][b] for b in bases)]
    out.update(resolved_both=res_both, resolved_any=res_any, informative_both=inf_both)
    if len({GROUPS[f] for f in res_both}) >= 2:
        return dict(out, outcome="LIMIT", name=NAMES["LIMIT"])
    if (len(inf_both) >= MIN_INFORMATIVE and all(lab[f][b] == "SUBSTANTIAL" for f in inf_both for b in bases)
            and not res_any and all(MB[b] >= MB_MIN for b in bases)):
        return dict(out, outcome="ROBUST", name=NAMES["ROBUST"])
    if res_any:
        return dict(out, outcome="SPECIFIC", name=NAMES["SPECIFIC"])
    return dict(out, outcome="INC", name=NAMES["INC"])
