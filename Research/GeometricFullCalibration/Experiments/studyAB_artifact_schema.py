"""
Frozen artifact schema and metric-bucket enforcement for Study A/B (§6, §9 of
BENCHMARK_IMPLEMENTATION_PLAN.md).

`_metric_bucket` is never inferred after the fact: every writer call must
pass an explicit bucket via `metric_bucket_for_method`, and a `scalar_only`
method's nll/brier/classwise fields are written as a real JSON `null`, never
an omitted key (so downstream aggregation fails loudly on a missing key
instead of silently treating it as zero) -- and, per validation test §10.7,
the writer raises rather than silently accepting a fabricated non-null value
for those fields on a scalar_only method.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional

SCALAR_ONLY = "scalar_only"
FULL_VECTOR = "full_vector"

# Frozen §6 table. Prefixes cover the Study-B cell family (studyB_cell_A/B/C/D)
# and the corrected-RGCL family without enumerating every draw-seed variant.
_EXACT_METHOD_BUCKETS: Dict[str, str] = {
    "uncalibrated": FULL_VECTOR,
    "base_model": FULL_VECTOR,
    "temperature_scaling": FULL_VECTOR,
    "parameterized_temperature_scaling": FULL_VECTOR,
    "vector_scaling": FULL_VECTOR,
    "odir_dirichlet": FULL_VECTOR,
    "native_dac": FULL_VECTOR,
    "kcal": FULL_VECTOR,
    "top_label_isotonic": SCALAR_ONLY,
    "gc_dac": SCALAR_ONLY,
    "published_rgcl": SCALAR_ONLY,
    "corrected_rgcl_studyA": SCALAR_ONLY,
    "trust_score_original_diagnostic": SCALAR_ONLY,
    "trust_score_original_switch": SCALAR_ONLY,
    "mahalanobis_confidence": SCALAR_ONLY,
    "full_vector_geometric_fusion": FULL_VECTOR,
    "glad_pi": FULL_VECTOR,
    "glad_pi_zero_geometry": FULL_VECTOR,
}
_PREFIX_METHOD_BUCKETS = (
    ("studyB_cell_", SCALAR_ONLY),
    ("trust_score", SCALAR_ONLY),
    ("contrastive_beta_", FULL_VECTOR),
)


def metric_bucket_for_method(method_name: str) -> str:
    if method_name in _EXACT_METHOD_BUCKETS:
        return _EXACT_METHOD_BUCKETS[method_name]
    for prefix, bucket in _PREFIX_METHOD_BUCKETS:
        if method_name.startswith(prefix):
            return bucket
    raise ValueError(
        f"Method '{method_name}' has no frozen §6 metric-bucket entry -- add it to "
        "_EXACT_METHOD_BUCKETS or _PREFIX_METHOD_BUCKETS before writing an artifact row "
        "for it (never infer the bucket after the fact)."
    )


def config_hash(config: Dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_artifact_row(
    *,
    method: str,
    checkpoint_seed: int,
    rgc_draw_seed: Optional[int],
    representation_strategy: str,
    statistic: str,
    mapper: str,
    dataset: str,
    corruption: Optional[str],
    severity: Optional[int],
    split: str,
    sample_ids_ref: str,
    sample_count: int,
    metrics: Dict[str, Optional[float]],
    provenance: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build one §9-schema row. `metrics` must already contain whatever the
    caller computed; this function only fills/validates `_metric_bucket` and
    enforces the scalar_only null-field rule -- it does not compute metrics
    itself.

    `sample_count` is an addition beyond the plan's own §9 JSON example (which
    only carries `sample_ids` as a ref string) -- required by the smoke-test
    verification checklist so every row is self-describing without needing to
    dereference `sample_ids_ref`.
    """
    bucket = metric_bucket_for_method(method)

    if bucket == SCALAR_ONLY:
        forbidden = ("nll", "brier")
        for key in forbidden:
            if metrics.get(key) is not None:
                raise ValueError(
                    f"Method '{method}' is scalar_only per the frozen §6 table but "
                    f"metrics['{key}'] is non-null ({metrics.get(key)!r}). No post-hoc "
                    "reconstruction rule is used for scalar-only methods -- fix the "
                    "caller instead of relaxing this check."
                )
            metrics[key] = None
        for key in list(metrics.keys()):
            if key.startswith("classwise") and metrics.get(key) is not None:
                raise ValueError(
                    f"Method '{method}' is scalar_only; classwise metric '{key}' must be null."
                )

    metrics = dict(metrics)
    metrics["_metric_bucket"] = bucket

    return {
        "method": method,
        "checkpoint_seed": int(checkpoint_seed),
        "rgc_draw_seed": None if rgc_draw_seed is None else int(rgc_draw_seed),
        "representation_strategy": representation_strategy,
        "statistic": statistic,
        "mapper": mapper,
        "dataset": dataset,
        "corruption": corruption,
        "severity": None if severity is None else int(severity),
        "split": split,
        "sample_ids": sample_ids_ref,
        "sample_count": int(sample_count),
        "metrics": metrics,
        "provenance": provenance,
    }
