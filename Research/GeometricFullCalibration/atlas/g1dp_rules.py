"""Frozen G1-DP classification and cross-family verdict (docs/g1_dp_spec.md sec. 6-8) as pure functions. Inputs in pp.

classify(est, ci) -> 'POS' | 'REV' | 'NULL' | 'UNC'
verdict(labels, valid) where labels[family][checkpoint] = class label, valid = set of families valid in BOTH checkpoints.
"""
M = 0.5          # material scale (Stage-0 convention)
SHUF = 0.2       # shuffle-control protocol-fault threshold (Stage-0 convention)
MIN_VALID = 5
NAMES = {"ROBUST+": "CROSS-FAMILY ROBUST (+)", "ROBUST-": "CROSS-FAMILY ROBUST (-)", "SPECIFIC": "FAMILY-SPECIFIC",
         "NEGATIVE": "PANEL-NEGATIVE", "INC_VALID": "INCONCLUSIVE (validity)", "INC": "INCONCLUSIVE"}


def classify(est, ci):
    lo, hi = ci
    if est >= M and lo > 0:
        return "POS"
    if est <= -M and hi < 0:
        return "REV"
    if lo > -M and hi < M:
        return "NULL"
    return "UNC"


def verdict(labels, valid, checkpoints=(2, 4)):
    V = sorted(valid)
    if len(V) < MIN_VALID:
        return {"outcome": "INC_VALID", "name": NAMES["INC_VALID"], "valid_families": V}
    both = lambda f, lab: all(labels[f][c] == lab for c in checkpoints)          # noqa: E731
    anyc = lambda f, lab: any(labels[f][c] == lab for c in checkpoints)          # noqa: E731
    pos = [f for f in V if both(f, "POS")]; rev = [f for f in V if both(f, "REV")]; null = [f for f in V if both(f, "NULL")]
    out = {"valid_families": V, "pos_both": pos, "rev_both": rev, "null_both": null}
    if len(pos) >= 4 and not any(anyc(f, "REV") for f in V):
        return dict(out, outcome="ROBUST+", name=NAMES["ROBUST+"])
    if len(rev) >= 4 and not any(anyc(f, "POS") for f in V):
        return dict(out, outcome="ROBUST-", name=NAMES["ROBUST-"])
    if (1 <= len(pos) <= 3 or 1 <= len(rev) <= 3) and len(null) >= 2:
        return dict(out, outcome="SPECIFIC", name=NAMES["SPECIFIC"])
    if not any(anyc(f, "POS") or anyc(f, "REV") for f in V) and len(null) >= 4:
        return dict(out, outcome="NEGATIVE", name=NAMES["NEGATIVE"])
    return dict(out, outcome="INC", name=NAMES["INC"])


def family_valid(v):
    """v: dict of per-checkpoint validity booleans for one family and checkpoint."""
    return all(v.values())
