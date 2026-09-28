"""Frozen G1 decision rule (docs/g1_conditional_access_spec.md sec. 6) as a pure function. All inputs in percentage points.

Per checkpoint, `q` holds point estimates `x` and 95% intervals `x_ci` = (lo, hi) for:
  dcond8 (D-C at 8k), dcond25 (D-C at 2.5k), dE (D-E), dDF (D-F), s (|G-C|), nest (D-B at 8k),
  budget_drop (dcond25 - dcond8), catchup ((C-B)8k - (C-B)2.5k), and accuracies accB, accC, accG (8k, %).
Validity inputs: v_extract, v_converged, v_edge (bool, True = passes), shuf (D-shuf - C, fold-0 point estimate).
"""
M, N = 0.5, 0.2
LABEL_TO_OUTCOME = {"acc": "A", "neg": "B", "cap": "C", "div": "D"}
NAMES = {"A": "MATERIAL CONDITIONAL ACCESSIBILITY", "B": "PENULTIMATE SUFFICIENCY / NO MEANINGFUL CONDITIONAL INCREMENT",
         "C": "SAMPLE-EFFICIENCY / CAPACITY CONFOUND", "D": "GENERIC-DIVERSITY RESULT", "E": "MIXED / INCONCLUSIVE", "F": "INVALID"}


def valid(v: dict) -> list:
    fails = []
    if not v["v_extract"]:
        fails.append("V1 extraction consistency")
    if not v["v_converged"]:
        fails.append("V2 unconverged final fit")
    if not v["v_edge"]:
        fails.append("V3 unresolved lambda edge in C or D (>= 2 of 5 folds)")
    if not v["shuf"] < N:
        fails.append("V4 shuffled-P conditional control >= +0.2 pp")
    return fails


def mat(q):
    return q["dcond8"] >= M and q["dcond8_ci"][0] > 0 and q["dcond8"] > q["s"]


def neg(q):
    return q["dcond8_ci"][1] < N


def label(q: dict) -> str:
    is_mat = mat(q)
    e3 = (q["budget_drop"] >= M and q["budget_drop_ci"][0] > 0 and q["catchup"] >= M and q["catchup_ci"][0] > 0 and not is_mat)
    if q["nest"] < -M or e3 or (is_mat and not (q["dE"] >= M and q["dE_ci"][0] > 0)):
        return "cap"
    if is_mat and q["dDF_ci"][1] < M:
        return "div"
    if is_mat:
        return "acc"
    if neg(q) and max(q["accC"], q["accG"]) >= q["accB"] - M:
        return "neg"
    return "mid"


def decide(per_ckpt: dict, validity: dict) -> dict:
    fails = [f"checkpoint {s}: {f}" for s, v in validity.items() for f in valid(v)]
    if fails:
        return {"outcome": "F", "name": NAMES["F"], "labels": None, "reasons": fails}
    labels = {s: label(q) for s, q in per_ckpt.items()}
    ls = set(labels.values())
    out = LABEL_TO_OUTCOME[ls.pop()] if len(ls) == 1 and next(iter(labels.values())) in LABEL_TO_OUTCOME else "E"
    return {"outcome": out, "name": NAMES[out], "labels": labels, "reasons": []}
