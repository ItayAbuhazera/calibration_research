import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from utils.method_metadata import SEMANTIC_SCHEMA_VERSION

from Experiments.run_unified_benchmark import _args_fingerprint
from Experiments.run_unified_benchmark import (
    _is_method_checkpoint_complete,
    _late_anchor_intermediate_paths,
    _reuse_non_metric_outputs,
    _save_method_checkpoint,
)
from Experiments.run_unified_benchmark_sbatch import (
    build_jobs,
    generate_sbatch_command,
    parse_args,
    validate_args,
)


def _launcher_args(tmp_path: Path, metric: str, output_template: str):
    if metric != "l2":
        l2_dir = Path(
            output_template.format(
                seed=11,
                model="resnet18",
                dataset="cifar10",
                method="baseline_cross_entropy",
                stab_metric="l2",
            )
        )
        l2_dir.mkdir(parents=True, exist_ok=True)
    args = parse_args(
        [
            "--dataset",
            "cifar10",
            "--model",
            "resnet18",
            "--seeds",
            "11",
            "--output-template",
            output_template,
            "--log-dir",
            str(tmp_path / "logs"),
            "--benchmark-extra-args",
            f"--stab_metric {metric}",
        ]
    )
    validate_args(args)
    return args


def test_whitened_cosine_jobs_do_not_reuse_legacy_l2_output(tmp_path):
    template = str(tmp_path / "benchmark_seed{seed}")
    l2_args = _launcher_args(tmp_path, "l2", template)
    whitened_args = _launcher_args(tmp_path, "whitened_cosine", template)

    l2_job = build_jobs(l2_args)[0]
    whitened_job = build_jobs(whitened_args)[0]

    assert l2_job.output_dir == tmp_path / "benchmark_seed11"
    assert whitened_job.output_dir == tmp_path / "benchmark_seed11_whitened_cosine"
    assert whitened_job.output_dir != l2_job.output_dir
    assert whitened_job.reuse_non_metric_from == l2_job.output_dir
    assert "ubm_resnet18_cifar10_s11_wcos" in generate_sbatch_command(
        whitened_job, whitened_args
    )
    assert "--reuse_non_metric_from" in generate_sbatch_command(
        whitened_job, whitened_args
    )


def test_metric_placeholder_controls_output_namespace(tmp_path):
    template = str(tmp_path / "{stab_metric}" / "benchmark_seed{seed}")
    args = _launcher_args(tmp_path, "whitened_cosine", template)

    job = build_jobs(args)[0]

    assert job.output_dir == tmp_path / "whitened_cosine" / "benchmark_seed11"
    assert job.reuse_non_metric_from == tmp_path / "l2" / "benchmark_seed11"


def test_resume_fingerprint_separates_whitened_cosine_from_legacy_l2():
    base = {
        "dataset": "cifar10",
        "model": "resnet18",
        "method": "baseline_cross_entropy",
        "seed": 11,
        "batch_size": 128,
        "target_dimension": 256,
        "num_layers": 6,
        "num_coordinates": 256,
        "enable_post_fusion_temperature": False,
        "enable_rgcl_tail_hybrids": False,
        "rgcl_tail_sources": "base",
    }
    legacy_l2 = _args_fingerprint(SimpleNamespace(**base))
    explicit_l2 = _args_fingerprint(SimpleNamespace(**base, stab_metric="l2"))
    whitened = _args_fingerprint(
        SimpleNamespace(
            **base,
            stab_metric="whitened_cosine",
            whitening_components=128,
            whitening_eps=1e-6,
        )
    )

    assert explicit_l2 == legacy_l2
    assert whitened != legacy_l2


def test_reuse_imports_invariant_checkpoints_but_not_metric_dependent_ones(tmp_path):
    source = tmp_path / "l2"
    target = tmp_path / "whitened_cosine"
    source_early = source / "intermediates" / "early"
    source_early.mkdir(parents=True)
    arrays = {
        "base_probs_test": np.array([[0.8, 0.2], [0.3, 0.7]]),
        "gc_dac_probs": np.array([[0.75, 0.25], [0.2, 0.8]]),
        "gc_dac_top_idx_test": np.array([0, 1]),
        "gc_dac_anchor_ct_test": np.array([0.75, 0.8]),
        "logits_val": np.zeros((2, 2)),
        "logits_test": np.zeros((2, 2)),
    }
    for name, value in arrays.items():
        np.save(source_early / f"{name}.npy", value)

    _save_method_checkpoint(
        str(source),
        "gc_dac",
        {"method_name": "gc_dac", "metrics": {}, "semantic_schema_version": SEMANTIC_SCHEMA_VERSION},
        arrays["gc_dac_probs"],
        False,
        "stage2_early",
    )
    _save_method_checkpoint(
        str(source),
        "full_vector_distance_fusion",
        {"method_name": "full_vector_distance_fusion", "metrics": {},
         "semantic_schema_version": SEMANTIC_SCHEMA_VERSION},
        arrays["base_probs_test"],
        True,
        "stage4_late_outputs",
    )
    source_anchor_paths = _late_anchor_intermediate_paths(str(source))
    np.save(source_anchor_paths["val_anchor_c_t"], np.array([0.6, 0.7]))
    Path(source_anchor_paths["val_anchor_diag"]).write_text(
        json.dumps({"folds": 5}), encoding="utf-8"
    )

    target_early = target / "intermediates" / "early"
    target_late = target / "intermediates" / "late"
    stage2_files = {
        **{
            name: str(target_early / f"{name}.npy")
            for name in arrays
        },
        "methods_so_far": str(target_late / "methods_so_far.json"),
        "method_probs_by_name": str(target_late / "method_probs_by_name.npz"),
        "method_can_change_argmax": str(
            target_late / "method_can_change_argmax.json"
        ),
    }
    progress = {"completed_stages": [], "stages": {}}

    reused = _reuse_non_metric_outputs(
        str(source), str(target), stage2_files, progress, 2, 2
    )

    assert reused == ["gc_dac"]
    assert _is_method_checkpoint_complete(str(target), "gc_dac", 2, 2)
    assert not _is_method_checkpoint_complete(
        str(target), "full_vector_distance_fusion", 2, 2
    )
    assert "stage2_early" in progress["completed_stages"]
    target_anchor_paths = _late_anchor_intermediate_paths(str(target))
    assert Path(target_anchor_paths["val_anchor_c_t"]).is_file()
