"""
Paired / common-seed analysis of CIFAR-100 rankgeom-mix results.

Compares contrastive-beta calibration methods against standard baselines on
the exact seeds shared by both methods.  Produces CSV tables and a Markdown
report under results/aggregated/paired_common_seed_analysis/.

Run from the repo root:
    python analysis/paired_common_seed_analysis.py
"""

import os
import math
import warnings
import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not found – Wilcoxon p-values and t-interval CIs will be NaN.")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILES = {
    "resnet18":  os.path.join(REPO_ROOT, "results/aggregated/c100_rankgeom_mix_per_seed_resnet18.csv"),
    "resnet152": os.path.join(REPO_ROOT, "results/aggregated/c100_rankgeom_mix_per_seed_resnet152.csv"),
    "resnet101": os.path.join(REPO_ROOT, "results/aggregated/c100_rankgeom_mix_per_seed_resnet101.csv"),
}

OUTPUT_DIR = os.path.join(REPO_ROOT, "results/aggregated/paired_common_seed_analysis")

# Contrastive-beta focal methods
CB_METHODS = [
    "contrastive_beta_ts",
    "contrastive_beta_ts_post_temperature",
    "contrastive_beta_vs",
    "contrastive_beta_vs_post_temperature",
]

# Requested baselines with fallback names in parentheses
REQUESTED_BASELINES = [
    "base_model",
    "temperature_scaling",
    "vector_scaling",
    "matrix_scaling",       # may not exist
    "dirichlet_odir",       # alias: odir_dirichlet
    "ovr_isotonic",
    "pooled_isotonic",      # may not exist
    "post_fusion_topiso",
    "anchored_rankgeom_tail_mixture",
    "softmax_knn_blend_rgcl",  # alias: softmax_knn_blend
    "kcal_lite_rgcl",
]

# Name aliasing: requested name → actual CSV name
ALIAS = {
    "dirichlet_odir":       "odir_dirichlet",
    "softmax_knn_blend_rgcl": "softmax_knn_blend",
}

# Primary metrics (lower is better for nll/brier/ece; higher for accuracy)
METRICS = ["accuracy", "nll", "brier", "top_label_ece"]

# Degeneracy thresholds
DEG_ACC   = 0.10
DEG_NLL   = 3.0
DEG_BRIER = 0.80
DEG_NET_FLIPS = -1000

# Decision-improving tolerances
TOL_NLL   = 0.005
TOL_BRIER = 0.002
TOL_ECE   = 0.005

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_name(requested: str, available: set) -> str | None:
    """Return the actual column name or None if unavailable."""
    if requested in available:
        return requested
    alias = ALIAS.get(requested)
    if alias and alias in available:
        return alias
    return None


def confidence_interval_95(deltas: pd.Series):
    """Return (mean, std, se, ci_lo, ci_hi) from per-seed deltas."""
    n = deltas.dropna()
    if len(n) == 0:
        return (np.nan,) * 5
    mu = n.mean()
    std = n.std(ddof=1) if len(n) > 1 else np.nan
    se = std / math.sqrt(len(n)) if len(n) > 1 else np.nan
    if len(n) >= 2 and HAS_SCIPY and not np.isnan(std):
        t = scipy_stats.t.ppf(0.975, df=len(n) - 1)
        ci_lo, ci_hi = mu - t * se, mu + t * se
    else:
        ci_lo, ci_hi = np.nan, np.nan
    return mu, std, se, ci_lo, ci_hi


def wilcoxon_p(deltas: pd.Series) -> float:
    """Return Wilcoxon signed-rank p-value, or NaN if not applicable."""
    d = deltas.dropna()
    nonzero = d[d != 0.0]
    if not HAS_SCIPY or len(nonzero) < 3:
        return np.nan
    try:
        _, p = scipy_stats.wilcoxon(nonzero)
        return p
    except Exception:
        return np.nan


def sign_test(deltas: pd.Series, higher_is_better: bool = True):
    """Return (n_improved, n_tied, n_worsened) counts.

    For accuracy (higher_is_better=True): improved = delta > 0.
    For NLL/Brier/ECE (higher_is_better=False): improved = delta < 0.
    """
    d = deltas.dropna()
    if higher_is_better:
        return (d > 0).sum(), (d == 0).sum(), (d < 0).sum()
    else:
        return (d < 0).sum(), (d == 0).sum(), (d > 0).sum()


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_df(path: str, backbone: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["backbone"] = backbone
    # Ensure numeric types for key columns
    for col in METRICS + ["flip_to_correct_count", "flip_to_wrong_count", "net_flips"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Core paired-delta computation
# ---------------------------------------------------------------------------

def compute_paired_deltas(df: pd.DataFrame, backbone: str):
    """
    For every (method, reference) pair, compute per-seed deltas on common seeds.
    Returns a list of dicts, one per (method, reference) pair.
    """
    available = set(df["method_name"].unique())

    # Resolve baseline names
    baselines = []
    for req in REQUESTED_BASELINES:
        actual = resolve_name(req, available)
        if actual:
            baselines.append((req, actual))
        # If missing, skip silently (reported in coverage table)

    # All methods we want to compare: CB + baselines (for cross-comparison)
    focal_methods = sorted(available)

    rows = []
    for method_name in focal_methods:
        df_m = df[df["method_name"] == method_name]
        m_seeds = set(df_m["seed"].unique())

        for req_ref, ref_name in baselines:
            if ref_name == method_name:
                continue
            df_r = df[df["method_name"] == ref_name]
            r_seeds = set(df_r["seed"].unique())
            common = sorted(m_seeds & r_seeds)

            row = {
                "backbone": backbone,
                "method": method_name,
                "reference_method": req_ref,
                "reference_method_actual": ref_name,
                "n_common_seeds": len(common),
                "common_seeds": str(common),
            }

            if len(common) == 0:
                for metric in METRICS:
                    row[f"mean_{metric}_method"] = np.nan
                    row[f"mean_{metric}_reference"] = np.nan
                    row[f"delta_{metric}"] = np.nan
                    row[f"delta_{metric}_std"] = np.nan
                    row[f"delta_{metric}_se"] = np.nan
                    row[f"delta_{metric}_ci_lo"] = np.nan
                    row[f"delta_{metric}_ci_hi"] = np.nan
                    row[f"delta_{metric}_wilcoxon_p"] = np.nan
                    row[f"delta_{metric}_n_improved"] = np.nan
                    row[f"delta_{metric}_n_tied"] = np.nan
                    row[f"delta_{metric}_n_worsened"] = np.nan
                row["mean_net_flips"] = np.nan
                row["mean_flip_to_correct"] = np.nan
                row["mean_flip_to_wrong"] = np.nan
                rows.append(row)
                continue

            # Pivot to per-seed values
            m_indexed = df_m.set_index("seed")
            r_indexed = df_r.set_index("seed")

            # higher is better for accuracy; lower is better for the rest
            metric_direction = {"accuracy": True, "nll": False, "brier": False, "top_label_ece": False}

            for metric in METRICS:
                m_vals = m_indexed.loc[common, metric] if metric in m_indexed.columns else pd.Series([np.nan]*len(common))
                r_vals = r_indexed.loc[common, metric] if metric in r_indexed.columns else pd.Series([np.nan]*len(common))
                # sign: method - reference
                deltas = m_vals.values - r_vals.values
                deltas_s = pd.Series(deltas)

                mu, std, se, ci_lo, ci_hi = confidence_interval_95(deltas_s)
                w_p = wilcoxon_p(deltas_s)
                hib = metric_direction.get(metric, True)
                n_imp, n_tie, n_wor = sign_test(deltas_s, higher_is_better=hib)

                row[f"mean_{metric}_method"] = float(np.nanmean(m_vals))
                row[f"mean_{metric}_reference"] = float(np.nanmean(r_vals))
                row[f"delta_{metric}"] = mu
                row[f"delta_{metric}_std"] = std
                row[f"delta_{metric}_se"] = se
                row[f"delta_{metric}_ci_lo"] = ci_lo
                row[f"delta_{metric}_ci_hi"] = ci_hi
                row[f"delta_{metric}_wilcoxon_p"] = w_p
                row[f"delta_{metric}_n_improved"] = n_imp
                row[f"delta_{metric}_n_tied"] = n_tie
                row[f"delta_{metric}_n_worsened"] = n_wor

            # Flip counts (vs base_model only via the CSV columns)
            if "net_flips" in m_indexed.columns:
                row["mean_net_flips"] = float(np.nanmean(m_indexed.loc[common, "net_flips"]))
            else:
                row["mean_net_flips"] = np.nan
            if "flip_to_correct_count" in m_indexed.columns:
                row["mean_flip_to_correct"] = float(np.nanmean(m_indexed.loc[common, "flip_to_correct_count"]))
            else:
                row["mean_flip_to_correct"] = np.nan
            if "flip_to_wrong_count" in m_indexed.columns:
                row["mean_flip_to_wrong"] = float(np.nanmean(m_indexed.loc[common, "flip_to_wrong_count"]))
            else:
                row["mean_flip_to_wrong"] = np.nan

            rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Seed-coverage table
# ---------------------------------------------------------------------------

def compute_seed_coverage(df: pd.DataFrame, backbone: str) -> pd.DataFrame:
    available = set(df["method_name"].unique())
    all_seeds = set(df["seed"].unique())

    ref_methods = {"base_model", "temperature_scaling", "vector_scaling"}
    ref_seed_sets = {}
    for r in ref_methods:
        actual = resolve_name(r, available)
        if actual:
            ref_seed_sets[r] = set(df[df["method_name"] == actual]["seed"].unique())

    rows = []
    for method in sorted(available):
        m_seeds = set(df[df["method_name"] == method]["seed"].unique())
        r = {"backbone": backbone, "method": method, "n_seeds": len(m_seeds), "seeds": str(sorted(m_seeds))}
        for ref, rs in ref_seed_sets.items():
            common = m_seeds & rs
            r[f"n_common_with_{ref}"] = len(common)
            r[f"common_seeds_{ref}"] = str(sorted(common))
        rows.append(r)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Degeneracy detection
# ---------------------------------------------------------------------------

def detect_degenerate(df: pd.DataFrame, backbone: str) -> pd.DataFrame:
    """Flag methods that look pathological on their available seeds."""
    rows = []
    for method in sorted(df["method_name"].unique()):
        sub = df[df["method_name"] == method]
        flags = []
        mean_acc   = sub["accuracy"].mean()
        mean_nll   = sub["nll"].mean()
        mean_brier = sub["brier"].mean()
        mean_netfl = sub["net_flips"].mean() if "net_flips" in sub.columns else np.nan

        if mean_acc < DEG_ACC:
            flags.append(f"accuracy={mean_acc:.3f}<{DEG_ACC}")
        if mean_nll > DEG_NLL:
            flags.append(f"nll={mean_nll:.3f}>{DEG_NLL}")
        if mean_brier > DEG_BRIER:
            flags.append(f"brier={mean_brier:.3f}>{DEG_BRIER}")
        if not np.isnan(mean_netfl) and mean_netfl < DEG_NET_FLIPS:
            flags.append(f"net_flips={mean_netfl:.0f}<{DEG_NET_FLIPS}")

        rows.append({
            "backbone": backbone,
            "method": method,
            "mean_accuracy": mean_acc,
            "mean_nll": mean_nll,
            "mean_brier": mean_brier,
            "mean_net_flips": mean_netfl,
            "is_degenerate": len(flags) > 0,
            "degenerate_flags": "; ".join(flags) if flags else "",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Decision-improving scorecard
# ---------------------------------------------------------------------------

def build_scorecard(deltas_df: pd.DataFrame, backbone: str) -> pd.DataFrame:
    cb_rows = deltas_df[
        (deltas_df["backbone"] == backbone) &
        (deltas_df["method"].isin(CB_METHODS))
    ]
    rows = []
    for _, r in cb_rows.iterrows():
        ref = r["reference_method"]
        n = r["n_common_seeds"]
        if n == 0:
            continue
        row = {
            "backbone": backbone,
            "method": r["method"],
            "reference_method": ref,
            "n_common_seeds": n,
        }
        # Accuracy: positive delta is good
        da = r.get("delta_accuracy", np.nan)
        row["accuracy_delta_gt0"] = bool(da > 0) if not np.isnan(da) else None
        row["delta_accuracy"] = da

        # Net flips: positive is good
        nf = r.get("mean_net_flips", np.nan)
        row["net_flips_positive"] = bool(nf > 0) if not np.isnan(nf) else None
        row["mean_net_flips"] = nf

        # NLL: delta <= 0 or within tolerance
        dn = r.get("delta_nll", np.nan)
        row["nll_ok"] = bool(dn <= TOL_NLL) if not np.isnan(dn) else None
        row["delta_nll"] = dn

        # Brier: delta <= 0 or within tolerance
        db = r.get("delta_brier", np.nan)
        row["brier_ok"] = bool(db <= TOL_BRIER) if not np.isnan(db) else None
        row["delta_brier"] = db

        # ECE: delta <= 0 or within tolerance
        de = r.get("delta_top_label_ece", np.nan)
        row["ece_ok"] = bool(de <= TOL_ECE) if not np.isnan(de) else None
        row["delta_ece"] = de

        # Overall decision-improving: acc up AND net_flips positive
        all_ok = (
            row["accuracy_delta_gt0"] is True
            and row["net_flips_positive"] is True
        )
        row["decision_improving"] = all_ok
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def fmt(v, decimals=4):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{v:.{decimals}f}"


def build_markdown_report(
    all_deltas: pd.DataFrame,
    cb_deltas: pd.DataFrame,
    coverage: pd.DataFrame,
    scorecard: pd.DataFrame,
    degenerate: pd.DataFrame,
) -> str:
    lines = []
    lines.append("# Paired / Common-Seed Analysis Report")
    lines.append("")
    lines.append("**Dataset:** CIFAR-100 (rankgeom-mix runs)")
    lines.append("")

    # ---- Executive Summary ----
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(
        "This report compares contrastive-beta calibration methods against standard baselines "
        "**on the exact seeds common to both**, to avoid unpaired-mean bias.  "
        "Contrastive-beta methods include `contrastive_beta_ts`, "
        "`contrastive_beta_ts_post_temperature`, `contrastive_beta_vs`, and "
        "`contrastive_beta_vs_post_temperature`."
    )
    lines.append("")
    lines.append(
        "Sign convention for deltas: `delta = method − reference`.  "
        "Positive `delta_accuracy` and `delta_net_flips` are desirable; "
        "negative `delta_nll`, `delta_brier`, `delta_ece` are desirable."
    )
    lines.append("")

    # ---- Per-backbone sections ----
    for backbone in ["resnet18", "resnet152", "resnet101"]:
        lines.append(f"---")
        lines.append(f"## Backbone: {backbone}")
        lines.append("")

        # Seed coverage
        cov = coverage[coverage["backbone"] == backbone]
        lines.append("### Seed Coverage")
        lines.append("")
        lines.append("| Method | n_seeds | n_common/base_model | n_common/temperature_scaling | n_common/vector_scaling |")
        lines.append("|--------|---------|---------------------|------------------------------|------------------------|")
        cb_cov = cov[cov["method"].isin(CB_METHODS + [
            "base_model", "temperature_scaling", "vector_scaling",
            "ovr_isotonic", "post_fusion_topiso", "anchored_rankgeom_tail_mixture",
            "softmax_knn_blend", "kcal_lite_rgcl",
        ])]
        for _, row in cb_cov.sort_values("method").iterrows():
            n_bm  = row.get("n_common_with_base_model", "N/A")
            n_ts  = row.get("n_common_with_temperature_scaling", "N/A")
            n_vs  = row.get("n_common_with_vector_scaling", "N/A")
            lines.append(f"| {row['method']} | {row['n_seeds']} | {n_bm} | {n_ts} | {n_vs} |")
        lines.append("")

        # Contrastive-beta paired results vs each reference
        bd = cb_deltas[cb_deltas["backbone"] == backbone]
        if bd.empty:
            lines.append("_No contrastive-beta results found for this backbone._")
            lines.append("")
        else:
            lines.append("### Paired Deltas: Contrastive-Beta vs Baselines")
            lines.append("")
            for ref in ["base_model", "temperature_scaling", "vector_scaling"]:
                sub = bd[bd["reference_method"] == ref]
                if sub.empty:
                    continue
                lines.append(f"#### vs `{ref}`")
                lines.append("")
                lines.append(
                    "| Method | n | Δacc | Δnll | Δbrier | Δece | "
                    "Δacc 95%CI | net_flips | flip_correct | flip_wrong |"
                )
                lines.append("|--------|---|------|------|--------|------|------------|-----------|--------------|------------|")
                for _, r in sub.iterrows():
                    n = r["n_common_seeds"]
                    da = fmt(r.get("delta_accuracy"), 4)
                    dn = fmt(r.get("delta_nll"), 4)
                    db = fmt(r.get("delta_brier"), 4)
                    de = fmt(r.get("delta_top_label_ece"), 4)
                    ci = (
                        f"[{fmt(r.get('delta_accuracy_ci_lo'), 4)}, {fmt(r.get('delta_accuracy_ci_hi'), 4)}]"
                        if n >= 2 else "N/A"
                    )
                    nf  = fmt(r.get("mean_net_flips"), 1)
                    fc  = fmt(r.get("mean_flip_to_correct"), 1)
                    fw  = fmt(r.get("mean_flip_to_wrong"), 1)
                    lines.append(
                        f"| {r['method']} | {n} | {da} | {dn} | {db} | {de} | {ci} | {nf} | {fc} | {fw} |"
                    )
                lines.append("")

        # Scorecard
        sc = scorecard[scorecard["backbone"] == backbone]
        if not sc.empty:
            lines.append("### Decision-Improving Scorecard")
            lines.append("")
            lines.append(
                "Criteria: `delta_accuracy > 0`, `net_flips > 0`, "
                f"`delta_nll ≤ +{TOL_NLL}`, `delta_brier ≤ +{TOL_BRIER}`, `delta_ece ≤ +{TOL_ECE}`"
            )
            lines.append("")
            lines.append("| Method | Ref | n | acc↑ | net_flip↑ | nll_ok | brier_ok | ece_ok | decision_improving |")
            lines.append("|--------|-----|---|------|-----------|--------|----------|--------|--------------------|")
            for _, r in sc.iterrows():
                def fmt_bool(v):
                    if v is None:
                        return "N/A"
                    return "✓" if v else "✗"

                lines.append(
                    f"| {r['method']} | {r['reference_method']} | {r['n_common_seeds']} | "
                    f"{fmt_bool(r.get('accuracy_delta_gt0'))} | "
                    f"{fmt_bool(r.get('net_flips_positive'))} | "
                    f"{fmt_bool(r.get('nll_ok'))} | "
                    f"{fmt_bool(r.get('brier_ok'))} | "
                    f"{fmt_bool(r.get('ece_ok'))} | "
                    f"{fmt_bool(r.get('decision_improving'))} |"
                )
            lines.append("")

        # Best contrastive-beta by paired accuracy gain vs base
        base_sub = bd[bd["reference_method"] == "base_model"].copy() if not bd.empty else pd.DataFrame()
        if not base_sub.empty and base_sub["n_common_seeds"].max() > 0:
            base_sub = base_sub[base_sub["n_common_seeds"] > 0]
            best_row = base_sub.loc[base_sub["delta_accuracy"].fillna(-999).idxmax()]
            lines.append("### Best Contrastive-Beta Method (paired accuracy gain vs base_model)")
            lines.append("")
            lines.append(f"**Method:** `{best_row['method']}`  ")
            lines.append(f"**Common seeds with base_model:** {int(best_row['n_common_seeds'])}  ")
            lines.append(f"**Mean paired Δaccuracy:** {fmt(best_row.get('delta_accuracy'), 4)}  ")
            lines.append(f"**Mean net flips:** {fmt(best_row.get('mean_net_flips'), 1)}  ")
            lines.append(f"**Mean paired ΔNLL:** {fmt(best_row.get('delta_nll'), 4)}  ")
            lines.append(f"**Mean paired ΔBrier:** {fmt(best_row.get('delta_brier'), 4)}  ")
            lines.append(f"**Mean paired ΔECE:** {fmt(best_row.get('delta_top_label_ece'), 4)}  ")
            lines.append("")
            # Interpretation
            acc_up  = (not math.isnan(best_row.get("delta_accuracy", float("nan")))
                       and best_row["delta_accuracy"] > 0)
            nf_pos  = (not math.isnan(best_row.get("mean_net_flips", float("nan")))
                       and best_row["mean_net_flips"] > 0)
            nll_ok  = (not math.isnan(best_row.get("delta_nll", float("nan")))
                       and best_row["delta_nll"] <= TOL_NLL)
            brier_ok = (not math.isnan(best_row.get("delta_brier", float("nan")))
                        and best_row["delta_brier"] <= TOL_BRIER)
            ece_ok  = (not math.isnan(best_row.get("delta_top_label_ece", float("nan")))
                       and best_row["delta_top_label_ece"] <= TOL_ECE)
            summary_parts = []
            if acc_up:
                summary_parts.append("accuracy improves")
            else:
                summary_parts.append("accuracy does NOT improve")
            if nf_pos:
                summary_parts.append("net flips are positive (decision-improving)")
            else:
                summary_parts.append("net flips are zero or negative")
            if nll_ok and brier_ok and ece_ok:
                summary_parts.append("NLL/Brier/ECE remain competitive")
            else:
                issues = []
                if not nll_ok:
                    issues.append("NLL increases beyond tolerance")
                if not brier_ok:
                    issues.append("Brier increases beyond tolerance")
                if not ece_ok:
                    issues.append("ECE increases beyond tolerance")
                summary_parts.append("; ".join(issues))
            n = int(best_row["n_common_seeds"])
            caveat = f"Based on {n} common seed(s). " + (
                "Small n — treat CIs with caution." if n < 5 else ""
            )
            lines.append("**Interpretation:** " + ". ".join(summary_parts) + ". " + caveat)
            lines.append("")

        # Degeneracy
        degen = degenerate[(degenerate["backbone"] == backbone) & degenerate["is_degenerate"]]
        if not degen.empty:
            lines.append("### Flagged Degenerate Methods")
            lines.append("")
            lines.append("| Method | Flags |")
            lines.append("|--------|-------|")
            for _, r in degen.iterrows():
                lines.append(f"| {r['method']} | {r['degenerate_flags']} |")
            lines.append("")

    # ---- Final Verdict ----
    lines.append("---")
    lines.append("## Final Verdict")
    lines.append("")

    # Count how many (backbone, method) combos are decision-improving vs base
    if not scorecard.empty:
        vs_base = scorecard[scorecard["reference_method"] == "base_model"]
        if not vs_base.empty:
            n_di = vs_base["decision_improving"].sum()
            total = len(vs_base)
            frac = n_di / total if total > 0 else 0
            # Per method summary
            method_summary = vs_base.groupby("method")["decision_improving"].agg(["sum", "count"])
            method_summary["frac"] = method_summary["sum"] / method_summary["count"]

            lines.append("### Summary across backbones (vs base_model)")
            lines.append("")
            lines.append("| CB Method | decision_improving (n/total backbone×seed combos) |")
            lines.append("|-----------|---------------------------------------------------|")
            for meth, row in method_summary.iterrows():
                lines.append(f"| {meth} | {int(row['sum'])}/{int(row['count'])} |")
            lines.append("")

            best_frac = method_summary["frac"].max()
            if best_frac >= 0.7:
                verdict = "**strong evidence**"
                justification = (
                    f"The best contrastive-beta variant achieves decision-improving criteria "
                    f"(Δacc > 0 and net_flips > 0) in {best_frac:.0%} of backbone cases."
                )
            elif best_frac >= 0.4:
                verdict = "**promising but incomplete**"
                justification = (
                    f"The best contrastive-beta variant achieves decision-improving criteria "
                    f"in {best_frac:.0%} of backbone cases — majority, but not consistent across all backbones."
                )
            else:
                verdict = "**not supported**"
                justification = (
                    f"Less than 40% of backbone cases show decision-improving behavior "
                    f"(best frac: {best_frac:.0%})."
                )
        else:
            verdict = "**not supported**"
            justification = "No vs-base_model comparisons found in the scorecard."
    else:
        verdict = "**not supported**"
        justification = "Scorecard is empty."

    lines.append(f"**Verdict:** {verdict}")
    lines.append("")
    lines.append(justification)
    lines.append("")
    lines.append(
        "> **Important caveat:** This analysis is limited to seeds where both the contrastive-beta "
        "method and the reference method produced results.  No claims are made about non-common seeds.  "
        "Small n (especially resnet152 with only 5 seeds) limits statistical power."
    )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_delta_rows = []
    all_coverage_dfs = []
    all_degen_dfs = []

    for backbone, path in INPUT_FILES.items():
        print(f"Processing {backbone} ...")
        df = load_df(path, backbone)

        delta_rows = compute_paired_deltas(df, backbone)
        all_delta_rows.extend(delta_rows)

        cov_df = compute_seed_coverage(df, backbone)
        all_coverage_dfs.append(cov_df)

        degen_df = detect_degenerate(df, backbone)
        all_degen_dfs.append(degen_df)

    all_deltas = pd.DataFrame(all_delta_rows)
    coverage   = pd.concat(all_coverage_dfs, ignore_index=True)
    degenerate = pd.concat(all_degen_dfs, ignore_index=True)

    # Contrastive-only subset
    cb_deltas = all_deltas[all_deltas["method"].isin(CB_METHODS)].copy()

    # Scorecard
    scorecard_parts = []
    for backbone in INPUT_FILES:
        sc = build_scorecard(all_deltas, backbone)
        scorecard_parts.append(sc)
    scorecard = pd.concat(scorecard_parts, ignore_index=True) if scorecard_parts else pd.DataFrame()

    # Save outputs
    paths = {}

    p = os.path.join(OUTPUT_DIR, "paired_deltas_all_backbones.csv")
    all_deltas.to_csv(p, index=False)
    paths["paired_deltas_all_backbones"] = p

    p = os.path.join(OUTPUT_DIR, "paired_deltas_contrastive_only.csv")
    cb_deltas.to_csv(p, index=False)
    paths["paired_deltas_contrastive_only"] = p

    p = os.path.join(OUTPUT_DIR, "method_seed_coverage.csv")
    coverage.to_csv(p, index=False)
    paths["method_seed_coverage"] = p

    p = os.path.join(OUTPUT_DIR, "decision_improving_scorecard.csv")
    scorecard.to_csv(p, index=False)
    paths["decision_improving_scorecard"] = p

    p = os.path.join(OUTPUT_DIR, "degenerate_method_flags.csv")
    degenerate.to_csv(p, index=False)
    paths["degenerate_method_flags"] = p

    # Build and save Markdown report
    md = build_markdown_report(all_deltas, cb_deltas, coverage, scorecard, degenerate)
    p = os.path.join(OUTPUT_DIR, "paired_common_seed_report.md")
    with open(p, "w") as f:
        f.write(md)
    paths["paired_common_seed_report"] = p

    print("\nGenerated output files:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
