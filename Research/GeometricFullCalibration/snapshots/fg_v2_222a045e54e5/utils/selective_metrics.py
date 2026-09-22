"""
Selective-prediction and confidence-ranking metrics.

ONE implementation shared by scalar-confidence and full-vector methods. The
caller supplies `(confidence, correct)`; how those two arrays were obtained is
the caller's responsibility and is fixed by the method's registry semantics:

  scalar_confidence methods:
      confidence = calibrated scalar assigned to the BASE predicted class
      correct    = (base_pred == y)
  full_vector methods:
      confidence = max(probs, axis=1)
      correct    = (argmax(probs) == y)

ORIENTATION (checked, not assumed): `confidence` is always "higher = more
likely correct". Every function here sorts DESCENDING by confidence and
treats `correct` as the positive class. A score with the opposite orientation
(e.g. a raw distance) must be negated by the caller before it gets here.

--------------------------------------------------------------------------
AURC CONVENTION (exact, so results are reproducible and tie-invariant)
--------------------------------------------------------------------------
Let the samples be sorted by decreasing confidence. For coverage
k/n (k = 1..n) the selective risk is

    risk(k) = (# errors among the k most-confident) / k

and

    AURC = (1/n) * sum_{k=1..n} risk(k)

i.e. the unweighted mean of the risk at every achievable coverage level,
which equals the trapezoid-free left-Riemann sum over the coverage grid
{1/n, ..., n/n}. This is the Geifman et al. convention.

TIES. Sorting alone leaves AURC dependent on the arbitrary order within a
block of equal confidences. We therefore replace each tied block's per-sample
correctness by the block MEAN before accumulating. That is exactly the
expected risk under uniform random tie-breaking, is deterministic, and is
invariant to the input order. Consequences worth knowing:
  * constant confidence  -> risk(k) = overall error rate for every k,
                            so AURC = error rate exactly;
  * all correct          -> AURC = 0;
  * all incorrect        -> AURC = 1.

`coverage_at_risk(target)` returns the LARGEST coverage k/n whose risk(k) is
<= target, or 0.0 when no coverage level achieves it (including the case where
even the single most-confident sample is an error). Because risk(k) is not
monotone in k, we scan all k rather than binary-searching.

`correctness_auroc` is the AUROC of `confidence` as a score for the positive
class `correct`. It is undefined (returned as None) when every sample is
correct or every sample is wrong.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# Risk levels at which coverage is reported by default. These are fixed in
# code (not tuned per run or per corruption cell) so no method can pick a
# favourable operating point after seeing results.
DEFAULT_TARGET_RISKS: tuple = (0.01, 0.02, 0.05, 0.10, 0.20)


def _validate(confidence: np.ndarray, correct: np.ndarray) -> tuple:
    confidence = np.asarray(confidence, dtype=np.float64).reshape(-1)
    correct = np.asarray(correct).reshape(-1)
    if confidence.shape != correct.shape:
        raise ValueError(
            f"confidence {confidence.shape} and correct {correct.shape} must align"
        )
    if confidence.size == 0:
        raise ValueError("Empty confidence/correct arrays")
    if not np.isfinite(confidence).all():
        raise ValueError("confidence contains NaN or Inf")
    correct = correct.astype(np.float64)
    if not np.all((correct == 0.0) | (correct == 1.0)):
        raise ValueError("correct must be boolean/0-1 valued")
    return confidence, correct


def tie_averaged_correct_by_confidence(
    confidence: np.ndarray, correct: np.ndarray
) -> np.ndarray:
    """Correctness sorted by decreasing confidence, averaged within tied blocks.

    Returns the expected correctness sequence under uniform random
    tie-breaking -- the basis of every tie-invariant quantity in this module.
    """
    confidence, correct = _validate(confidence, correct)
    order = np.argsort(-confidence, kind="stable")
    conf_sorted = confidence[order]
    corr_sorted = correct[order]

    # Block boundaries of equal confidence.
    new_block = np.empty(conf_sorted.shape, dtype=bool)
    new_block[0] = True
    new_block[1:] = conf_sorted[1:] != conf_sorted[:-1]
    block_id = np.cumsum(new_block) - 1
    block_sums = np.bincount(block_id, weights=corr_sorted)
    block_counts = np.bincount(block_id)
    return (block_sums / block_counts)[block_id]


def risk_coverage_curve(confidence: np.ndarray, correct: np.ndarray) -> Dict[str, np.ndarray]:
    """Coverage/risk curve at every achievable coverage level k/n."""
    corr_eff = tie_averaged_correct_by_confidence(confidence, correct)
    n = corr_eff.size
    k = np.arange(1, n + 1, dtype=np.float64)
    cumulative_errors = np.cumsum(1.0 - corr_eff)
    return {"coverage": k / n, "risk": cumulative_errors / k}


def aurc(confidence: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve; see the module docstring."""
    curve = risk_coverage_curve(confidence, correct)
    return float(np.mean(curve["risk"]))


def excess_aurc(confidence: np.ndarray, correct: np.ndarray) -> float:
    """AURC minus the AURC of a perfect confidence ranking (same error count).

    The optimal ranking puts every correct sample ahead of every error, so its
    AURC depends only on the error rate. Reporting the excess makes AURC
    comparable across cells with different accuracies, which matters because
    CIFAR-C cells have very different base accuracies.
    """
    _, correct_arr = _validate(confidence, correct)
    n = correct_arr.size
    n_err = int(round(float(np.sum(1.0 - correct_arr))))
    oracle = np.concatenate([np.ones(n - n_err), np.zeros(n_err)])
    k = np.arange(1, n + 1, dtype=np.float64)
    oracle_aurc = float(np.mean(np.cumsum(1.0 - oracle) / k))
    return float(aurc(confidence, correct) - oracle_aurc)


def correctness_auroc(confidence: np.ndarray, correct: np.ndarray) -> Optional[float]:
    """AUROC of `confidence` as a score for correctness. None if degenerate.

    Computed from ranks (Mann-Whitney U) with proper mid-rank handling for
    tied confidences, so it needs no sklearn and is tie-invariant.
    """
    confidence, correct = _validate(confidence, correct)
    n_pos = float(np.sum(correct))
    n_neg = float(correct.size) - n_pos
    if n_pos == 0.0 or n_neg == 0.0:
        return None
    order = np.argsort(confidence, kind="stable")
    sorted_conf = confidence[order]
    ranks = np.empty(confidence.size, dtype=np.float64)
    i = 0
    while i < sorted_conf.size:
        j = i
        while j + 1 < sorted_conf.size and sorted_conf[j + 1] == sorted_conf[i]:
            j += 1
        # Mid-rank (1-based) for the tied block [i, j].
        ranks[order[i : j + 1]] = 0.5 * ((i + 1) + (j + 1))
        i = j + 1
    sum_ranks_pos = float(np.sum(ranks[correct == 1.0]))
    return float((sum_ranks_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg))


def coverage_at_risk(
    confidence: np.ndarray, correct: np.ndarray, target_risk: float
) -> float:
    """Largest coverage whose selective risk is <= target_risk (0.0 if none)."""
    if not (0.0 <= target_risk <= 1.0):
        raise ValueError(f"target_risk must be in [0, 1], got {target_risk}")
    curve = risk_coverage_curve(confidence, correct)
    feasible = curve["risk"] <= target_risk + 1e-12
    if not np.any(feasible):
        return 0.0
    return float(curve["coverage"][np.max(np.flatnonzero(feasible))])


def risk_at_coverage(
    confidence: np.ndarray, correct: np.ndarray, target_coverage: float
) -> float:
    """Selective risk at the smallest achievable coverage >= target_coverage."""
    if not (0.0 < target_coverage <= 1.0):
        raise ValueError(f"target_coverage must be in (0, 1], got {target_coverage}")
    curve = risk_coverage_curve(confidence, correct)
    idx = int(np.searchsorted(curve["coverage"], target_coverage - 1e-12, side="left"))
    idx = min(idx, curve["coverage"].size - 1)
    return float(curve["risk"][idx])


def selective_metrics(
    confidence: np.ndarray,
    correct: np.ndarray,
    *,
    target_risks: Sequence[float] = DEFAULT_TARGET_RISKS,
    curve_points: int = 0,
) -> Dict[str, Any]:
    """All confidence-ranking / selective-prediction metrics for one method.

    `curve_points` > 0 additionally returns a subsampled risk-coverage curve
    (that many evenly spaced coverage levels) for plotting; 0 omits it so
    summary artifacts stay small.
    """
    confidence, correct = _validate(confidence, correct)
    out: Dict[str, Any] = {
        "correctness_auroc": correctness_auroc(confidence, correct),
        "aurc": aurc(confidence, correct),
        "excess_aurc": excess_aurc(confidence, correct),
    }
    # Flat keys (no nested dicts): summary rows are also written to CSV, and a
    # nested dict there would either be stringified or dropped.
    for r in target_risks:
        out[f"coverage_at_risk_{float(r):g}"] = coverage_at_risk(
            confidence, correct, float(r)
        )
    for c in (0.5, 0.7, 0.8, 0.9, 1.0):
        out[f"risk_at_coverage_{c:g}"] = risk_at_coverage(confidence, correct, c)
    if curve_points > 0:
        curve = risk_coverage_curve(confidence, correct)
        n = curve["coverage"].size
        idx = np.unique(
            np.linspace(0, n - 1, min(curve_points, n)).astype(int)
        )
        out["risk_coverage_curve_coverage"] = [float(v) for v in curve["coverage"][idx]]
        out["risk_coverage_curve_risk"] = [float(v) for v in curve["risk"][idx]]
    return out


def coverage_at_matched_risk(
    reference_confidence: np.ndarray,
    reference_correct: np.ndarray,
    candidate_confidence: np.ndarray,
    candidate_correct: np.ndarray,
    *,
    reference_coverage: float = 0.8,
) -> Dict[str, Any]:
    """Coverage the candidate reaches at the reference's risk at a given coverage.

    This is the head-to-head form of coverage-at-matched-risk the benchmark
    asks for: fix the operating risk from the reference method (the strongest
    non-geometric baseline), then ask how much coverage the candidate buys at
    that same risk. `delta_coverage` > 0 means the candidate serves more
    samples at equal risk.
    """
    matched_risk = risk_at_coverage(
        reference_confidence, reference_correct, reference_coverage
    )
    ref_cov = coverage_at_risk(reference_confidence, reference_correct, matched_risk)
    cand_cov = coverage_at_risk(candidate_confidence, candidate_correct, matched_risk)
    return {
        "reference_coverage": float(reference_coverage),
        "matched_risk": float(matched_risk),
        "reference_coverage_at_matched_risk": float(ref_cov),
        "candidate_coverage_at_matched_risk": float(cand_cov),
        "delta_coverage": float(cand_cov - ref_cov),
    }


def paired_complementarity(
    reference_confidence: np.ndarray,
    candidate_confidence: np.ndarray,
    correct: np.ndarray,
) -> Dict[str, Any]:
    """Per-sample complementarity diagnostic (DIAGNOSTIC ONLY -- never tuned on).

    Uses the per-sample confidence loss delta
        delta_i = loss_reference_i - loss_candidate_i,
    with loss_i the squared error of the confidence against correctness
    (the same object top-label ECE aggregates). A positive delta means the
    candidate assigned a better-calibrated confidence to sample i.

    `oracle_*` describe the unreachable upper bound from picking, per sample,
    whichever method happened to be better. A large oracle gain alongside a
    small realized gain is the signature of "complementary signal exists but
    current gating cannot identify it".
    """
    ref, corr = _validate(reference_confidence, correct)
    cand, _ = _validate(candidate_confidence, correct)
    loss_ref = (ref - corr) ** 2
    loss_cand = (cand - corr) ** 2
    delta = loss_ref - loss_cand
    oracle_loss = np.minimum(loss_ref, loss_cand)
    return {
        "n": int(corr.size),
        "mean_loss_reference": float(np.mean(loss_ref)),
        "mean_loss_candidate": float(np.mean(loss_cand)),
        "mean_delta": float(np.mean(delta)),
        "frac_candidate_better": float(np.mean(delta > 0)),
        "frac_reference_better": float(np.mean(delta < 0)),
        "frac_tied": float(np.mean(delta == 0)),
        "mean_oracle_loss": float(np.mean(oracle_loss)),
        "oracle_gain_over_best_single": float(
            min(np.mean(loss_ref), np.mean(loss_cand)) - np.mean(oracle_loss)
        ),
        "confidence_rank_correlation": _spearman(ref, cand),
        "note": "diagnostic only; must not be used for tuning or model selection",
    }


def _spearman(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Spearman rank correlation with mid-ranks; None if either side is constant."""
    ra, rb = _midranks(a), _midranks(b)
    sa, sb = float(np.std(ra)), float(np.std(rb))
    if sa == 0.0 or sb == 0.0:
        return None
    return float(np.mean((ra - np.mean(ra)) * (rb - np.mean(rb))) / (sa * sb))


def _midranks(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    order = np.argsort(x, kind="stable")
    xs = x[order]
    ranks = np.empty(x.size, dtype=np.float64)
    i = 0
    while i < xs.size:
        j = i
        while j + 1 < xs.size and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * ((i + 1) + (j + 1))
        i = j + 1
    return ranks
