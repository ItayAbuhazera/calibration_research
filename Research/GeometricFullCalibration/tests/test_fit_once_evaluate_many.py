"""
Tests for the fit-once/evaluate-many corruption-cell protocol added to
Experiments/run_unified_benchmark.py.

Required guarantee: for every checkpoint and every fitted method, fitting
happens exactly once on clean data; every --corruption_type run must load
that frozen state and must NEVER call the underlying .fit()/fit_callable.
These tests verify that guarantee directly (call-count assertions), verify
the hard-fail behavior when state is missing, and demonstrate item 1's
explicit requirement: for one checkpoint and at least two different
corruption cells, the fitted-calibrator artifact hash is identical.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from Experiments.run_unified_benchmark import (
    _fit_or_load_state,
    _fit_or_load_state_with_result,
    _fit_or_load_value,
    _fitted_state_dir_hash,
    _run_geo_cal_function_fit_once,
    _write_fit_once_provenance,
)


class _DummyCalibrator:
    """Stand-in for a real calibrator object (TemperatureScaling, VectorScaling,
    MahalanobisConfidenceCalibrator, GLADPICalibrator, ...): .fit() mutates
    instance state in place."""

    def __init__(self):
        self.fitted = False
        self.param = None

    def fit(self, value):
        self.fitted = True
        self.param = value
        return {"param": value}


def _args(tmp_path, corruption_type=None, corruption_severity=None, fitted_state_dir=None):
    return SimpleNamespace(
        corruption_type=corruption_type,
        corruption_severity=corruption_severity,
        fitted_state_dir=str(fitted_state_dir) if fitted_state_dir else None,
    )


# --------------------------------------------------------------------------
# _fit_or_load_state
# --------------------------------------------------------------------------
def test_clean_run_fits_and_persists(tmp_path):
    state_dir = tmp_path / "fitted"
    args = _args(tmp_path, fitted_state_dir=state_dir)
    obj = _DummyCalibrator()
    calls = []
    _fit_or_load_state(args, "dummy", obj, lambda: calls.append(1) or obj.fit(42))
    assert calls == [1]
    assert obj.fitted and obj.param == 42
    assert (state_dir / "dummy.pkl").exists()


def test_corruption_run_without_fitted_state_dir_raises(tmp_path):
    args = _args(tmp_path, corruption_type="fog", corruption_severity=3, fitted_state_dir=None)
    obj = _DummyCalibrator()
    with pytest.raises(RuntimeError, match="fitted_state_dir"):
        _fit_or_load_state(args, "dummy", obj, lambda: obj.fit(42))


def test_corruption_run_with_missing_pickle_raises_and_never_fits(tmp_path):
    state_dir = tmp_path / "fitted"
    state_dir.mkdir()
    args = _args(tmp_path, corruption_type="fog", corruption_severity=3, fitted_state_dir=state_dir)
    obj = _DummyCalibrator()
    calls = []
    with pytest.raises(RuntimeError, match="refusing to fit fresh"):
        _fit_or_load_state(args, "dummy", obj, lambda: calls.append(1) or obj.fit(42))
    assert calls == [], "fit_callable must NEVER be invoked on a corruption run with no fitted state"
    assert not obj.fitted


def test_corruption_run_loads_state_and_never_calls_fit_callable(tmp_path):
    state_dir = tmp_path / "fitted"
    clean_args = _args(tmp_path, fitted_state_dir=state_dir)
    obj = _DummyCalibrator()
    _fit_or_load_state(clean_args, "dummy", obj, lambda: obj.fit(99))

    corrupted_obj = _DummyCalibrator()
    calls = []
    corr_args = _args(tmp_path, corruption_type="gaussian_noise", corruption_severity=1, fitted_state_dir=state_dir)
    _fit_or_load_state(corr_args, "dummy", corrupted_obj, lambda: calls.append(1) or corrupted_obj.fit(-1))

    assert calls == [], "corruption run must never call fit_callable when state exists"
    assert corrupted_obj.fitted and corrupted_obj.param == 99, "restored state must match the clean fit, not a fresh one"


# --------------------------------------------------------------------------
# Item 1: identical fitted-artifact hash across >=2 different corruption cells
# --------------------------------------------------------------------------
def test_fitted_artifact_hash_identical_across_two_corruption_cells(tmp_path):
    state_dir = tmp_path / "fitted"
    clean_args = _args(tmp_path, fitted_state_dir=state_dir)
    obj = _DummyCalibrator()
    _fit_or_load_state(clean_args, "temperature_scaling", obj, lambda: obj.fit(1.37))

    hash_after_fit = _fitted_state_dir_hash(str(state_dir))
    assert hash_after_fit is not None

    cell_a_args = _args(tmp_path, corruption_type="gaussian_noise", corruption_severity=1, fitted_state_dir=state_dir)
    cell_b_args = _args(tmp_path, corruption_type="fog", corruption_severity=5, fitted_state_dir=state_dir)

    obj_a = _DummyCalibrator()
    calls_a = []
    _fit_or_load_state(cell_a_args, "temperature_scaling", obj_a, lambda: calls_a.append(1))
    hash_during_cell_a = _fitted_state_dir_hash(str(state_dir))

    obj_b = _DummyCalibrator()
    calls_b = []
    _fit_or_load_state(cell_b_args, "temperature_scaling", obj_b, lambda: calls_b.append(1))
    hash_during_cell_b = _fitted_state_dir_hash(str(state_dir))

    assert calls_a == [] and calls_b == [], "neither corruption cell may fit"
    assert hash_during_cell_a == hash_during_cell_b == hash_after_fit, (
        "the fitted-calibrator artifact hash must be identical across different "
        "corruption cells for the same checkpoint -- item 1's explicit requirement"
    )
    assert obj_a.param == obj_b.param == 1.37, "both cells must reuse the exact same fitted parameter"


def test_fitted_state_dir_hash_changes_when_contents_change(tmp_path):
    state_dir = tmp_path / "fitted"
    args = _args(tmp_path, fitted_state_dir=state_dir)
    obj1 = _DummyCalibrator()
    _fit_or_load_state(args, "method_a", obj1, lambda: obj1.fit(1))
    h1 = _fitted_state_dir_hash(str(state_dir))

    obj2 = _DummyCalibrator()
    _fit_or_load_state(args, "method_b", obj2, lambda: obj2.fit(2))
    h2 = _fitted_state_dir_hash(str(state_dir))

    assert h1 != h2


# --------------------------------------------------------------------------
# _fit_or_load_value (plain-value variant, e.g. trust_score's threshold dict)
# --------------------------------------------------------------------------
def test_fit_or_load_value_never_recomputes_on_corruption_run(tmp_path):
    state_dir = tmp_path / "fitted"
    clean_args = _args(tmp_path, fitted_state_dir=state_dir)
    calls = []

    def _select_threshold():
        calls.append(1)
        return {"threshold": 0.73}

    value = _fit_or_load_value(clean_args, "trust_score_original_switch", _select_threshold)
    assert value == {"threshold": 0.73}
    assert calls == [1]

    corr_args = _args(tmp_path, corruption_type="jpeg_compression", corruption_severity=1, fitted_state_dir=state_dir)
    value2 = _fit_or_load_value(corr_args, "trust_score_original_switch", lambda: calls.append(2))
    assert value2 == {"threshold": 0.73}
    assert calls == [1], "must not recompute the threshold on a corruption run"


# --------------------------------------------------------------------------
# _fit_or_load_state_with_result (GLAD-PI style: object state + return dict)
# --------------------------------------------------------------------------
def test_fit_or_load_state_with_result(tmp_path):
    state_dir = tmp_path / "fitted"
    clean_args = _args(tmp_path, fitted_state_dir=state_dir)
    obj = _DummyCalibrator()
    result = _fit_or_load_state_with_result(clean_args, "glad_pi", obj, lambda: obj.fit(5))
    assert result == {"param": 5}

    corr_args = _args(tmp_path, corruption_type="defocus_blur", corruption_severity=3, fitted_state_dir=state_dir)
    obj2 = _DummyCalibrator()
    calls = []
    result2 = _fit_or_load_state_with_result(
        corr_args, "glad_pi", obj2, lambda: calls.append(1) or obj2.fit(-99)
    )
    assert calls == []
    assert result2 == {"param": 5}
    assert obj2.param == 5


# --------------------------------------------------------------------------
# _run_geo_cal_function_fit_once (published RGCL / GC-DAC style)
# --------------------------------------------------------------------------
def test_run_geo_cal_function_fit_once(tmp_path):
    state_dir = tmp_path / "fitted"
    clean_args = _args(tmp_path, fitted_state_dir=state_dir)

    def fn(**kwargs):
        assert "precomputed_geo_cal" not in kwargs or kwargs.get("return_fitted_calibrator")
        return {"calibrated_probs": [1, 2, 3], "fitted_geo_cal": "the-fitted-object"}

    result = _run_geo_cal_function_fit_once(clean_args, "rgcl", fn, {"x": 1})
    assert result["calibrated_probs"] == [1, 2, 3]

    seen_kwargs = {}

    def fn_corruption(**kwargs):
        seen_kwargs.update(kwargs)
        assert "precomputed_geo_cal" in kwargs, "corruption run must receive the precomputed calibrator"
        return {"calibrated_probs": [4, 5, 6]}

    corr_args = _args(tmp_path, corruption_type="fog", corruption_severity=1, fitted_state_dir=state_dir)
    result2 = _run_geo_cal_function_fit_once(corr_args, "rgcl", fn_corruption, {"x": 1})
    assert result2["calibrated_probs"] == [4, 5, 6]
    assert seen_kwargs["precomputed_geo_cal"] == "the-fitted-object"


def test_run_geo_cal_function_fit_once_raises_without_fitted_state(tmp_path):
    state_dir = tmp_path / "fitted"
    state_dir.mkdir()
    corr_args = _args(tmp_path, corruption_type="fog", corruption_severity=1, fitted_state_dir=state_dir)

    def fn(**kwargs):
        raise AssertionError("fn must never be called with a real fit path on a corruption run with no state")

    with pytest.raises(RuntimeError, match="refusing to fit fresh"):
        _run_geo_cal_function_fit_once(corr_args, "rgcl", fn, {})


# --------------------------------------------------------------------------
# Item 4: structural assertion that corrupted data cannot reach fitting
# --------------------------------------------------------------------------
def test_corruption_loader_structurally_cannot_reach_fit_path(tmp_path):
    """
    Simulates the exact shape of a real call site: fit_callable closes over
    a 'corrupted_test_loader' sentinel that would only be used for evaluation
    (.calibrate()), never for .fit(). This asserts fit_callable is never
    invoked at all on a corruption run absent fitted state -- i.e. even if a
    caller mistakenly wired a corrupted loader into fit_callable's closure,
    it could never execute on a --corruption_type run without --fitted_state_dir
    already populated from a clean run.
    """
    poisoned_calls = []

    def fit_with_corrupted_data_by_mistake():
        poisoned_calls.append("corrupted_test_loader_touched_fit")
        raise AssertionError("this must never execute")

    args_no_state_dir = _args(tmp_path, corruption_type="fog", corruption_severity=3, fitted_state_dir=None)
    with pytest.raises(RuntimeError):
        _fit_or_load_state(args_no_state_dir, "m", _DummyCalibrator(), fit_with_corrupted_data_by_mistake)

    empty_state_dir = tmp_path / "empty_fitted"
    empty_state_dir.mkdir()
    args_empty_state_dir = _args(tmp_path, corruption_type="fog", corruption_severity=3, fitted_state_dir=empty_state_dir)
    with pytest.raises(RuntimeError):
        _fit_or_load_state(args_empty_state_dir, "m", _DummyCalibrator(), fit_with_corrupted_data_by_mistake)

    assert poisoned_calls == [], "corrupted data must never reach a fit call under any tested condition"


# --------------------------------------------------------------------------
# Item 3: provenance recording
# --------------------------------------------------------------------------
def test_write_fit_once_provenance_records_required_fields(tmp_path):
    state_dir = tmp_path / "fitted"
    state_dir.mkdir()
    obj = _DummyCalibrator()
    _fit_or_load_state(
        _args(tmp_path, fitted_state_dir=state_dir), "temperature_scaling", obj, lambda: obj.fit(1.0)
    )

    output_dir = tmp_path / "eval" / "fog_s3"
    args = SimpleNamespace(
        corruption_type="fog", corruption_severity=3, fitted_state_dir=str(state_dir),
        output_dir=str(output_dir), dataset="cifar100", model="resnet101", method="baseline_cross_entropy",
        seed=1, num_layers=6, target_dimension=256, batch_size=64,
    )
    _write_fit_once_provenance(args)

    import json
    with open(output_dir / "fit_once_provenance.json") as f:
        prov = json.load(f)

    assert prov["evaluation"]["corruption_type"] == "fog"
    assert prov["evaluation"]["corruption_severity"] == 3
    assert prov["fitted_state_dir"] == str(state_dir)
    assert prov["fitted_state_hash"] is not None
    assert prov["config_hash"]
    assert prov["clean_fitting_split"]["checkpoint_seed"] == 1


# --------------------------------------------------------------------------
# Regression: job 21392517 UnboundLocalError on native_dac_train_feats
# --------------------------------------------------------------------------
def _main_scope_bindings_and_deletes():
    """Names bound / `del`-ed directly in main()'s own scope (nested function
    and lambda bodies excluded -- their locals are not main's locals)."""
    import ast

    src = (Path(__file__).resolve().parents[1] / "Experiments" / "run_unified_benchmark.py").read_text()
    main_fn = next(
        n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    bound, deleted = set(), []

    def visit(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                if not isinstance(child, ast.Lambda):
                    bound.add(child.name)
                continue  # do not descend into nested scopes
            if isinstance(child, ast.Name):
                if isinstance(child.ctx, ast.Store):
                    bound.add(child.id)
                elif isinstance(child.ctx, ast.Del):
                    deleted.append((child.id, child.lineno))
            elif isinstance(child, (ast.Import, ast.ImportFrom)):
                for a in child.names:
                    bound.add((a.asname or a.name).split(".")[0])
            elif isinstance(child, ast.ExceptHandler) and child.name:
                bound.add(child.name)
            visit(child)

    for a in main_fn.args.args:
        bound.add(a.arg)
    visit(main_fn)
    return bound, deleted


def test_main_never_deletes_a_name_it_never_binds():
    """The fit-once refactor moved native-DAC train/val extraction into the
    deferred _fit_native_dac closure, but left `del native_dac_train_feats,
    native_dac_val_feats` in main()'s scope -> UnboundLocalError on every
    clean fit. Every `del name` in main() must name something main() binds."""
    bound, deleted = _main_scope_bindings_and_deletes()
    stale = [(n, ln) for n, ln in deleted if n not in bound]
    assert not stale, f"main() deletes names it never binds (stale cleanup): {stale}"


def test_native_dac_fit_features_stay_inside_deferred_closure():
    """Native-DAC train/val features are extracted only inside the deferred
    fit closure (so corruption runs, which load frozen state, never pay for
    or refit on them); main()'s scope must not reference them at all."""
    bound, deleted = _main_scope_bindings_and_deletes()
    for name in ("native_dac_train_feats", "native_dac_val_feats"):
        assert name not in bound
        assert name not in {n for n, _ in deleted}
