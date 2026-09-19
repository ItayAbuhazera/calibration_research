"""
Copy the 'kcal' method entry from a KCal-only run directory into an existing
benchmark run directory, then rebuild summary_metrics.json/.csv.

Usage:
    python Experiments/merge_kcal_into_run.py \
        --src_dir results/.../seed22/decision_run_v1_kcal \
        --dst_dir results/.../seed22/decision_run_v1
"""

import argparse
import json
import os
import shutil
import sys
import tempfile


SRC_METHOD_NAME = "kcal"  # name produced by run_unified_benchmark.py


def _dst_method_name(src_method_dir: str) -> str:
    """Derive the destination method name from the entry's projection_dim."""
    entry_path = os.path.join(src_method_dir, "entry.json")
    try:
        with open(entry_path) as f:
            entry = json.load(f)
        dim = entry.get("projection_dim")
        if dim is not None:
            return f"kcal_d{int(dim)}"
    except Exception:
        pass
    return SRC_METHOD_NAME


def _method_store_dir(output_dir: str) -> str:
    return os.path.join(output_dir, "intermediates", "method_outputs")


def _method_registry_path(output_dir: str) -> str:
    return os.path.join(_method_store_dir(output_dir), "method_registry.json")


def _load_json(path: str):
    with open(path) as f:
        return json.load(f)


def _atomic_write_json(path: str, data) -> None:
    dir_ = os.path.dirname(path)
    os.makedirs(dir_, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def merge(src_dir: str, dst_dir: str) -> None:
    src_method_dir = os.path.join(_method_store_dir(src_dir), SRC_METHOD_NAME)
    method_name = _dst_method_name(src_method_dir)
    dst_method_dir = os.path.join(_method_store_dir(dst_dir), method_name)
    print(f"  Destination method name: {method_name}")

    # Validate source
    for fname in ("entry.json", "probs.npy", "meta.json"):
        p = os.path.join(src_method_dir, fname)
        if not os.path.exists(p):
            sys.exit(f"ERROR: source file missing: {p}")

    if os.path.exists(os.path.join(dst_method_dir, "entry.json")):
        print(f"  '{method_name}' entry already present in {dst_dir}, overwriting.")

    # Copy method files
    os.makedirs(dst_method_dir, exist_ok=True)
    for fname in ("entry.json", "probs.npy", "meta.json"):
        shutil.copy2(
            os.path.join(src_method_dir, fname),
            os.path.join(dst_method_dir, fname),
        )
    print(f"  Copied {method_name} method files to {dst_method_dir}")

    # Update destination method registry
    dst_registry_path = _method_registry_path(dst_dir)
    registry = _load_json(dst_registry_path) if os.path.exists(dst_registry_path) else {}

    ordered = list(registry.get("ordered_methods", []))
    if method_name not in ordered:
        ordered.append(method_name)

    method_paths = dict(registry.get("method_paths", {}))
    method_paths[method_name] = {
        "entry": os.path.join(dst_method_dir, "entry.json"),
        "probs": os.path.join(dst_method_dir, "probs.npy"),
        "meta":  os.path.join(dst_method_dir, "meta.json"),
    }

    # Preserve can_change_argmax and stage from source registry
    src_registry = _load_json(_method_registry_path(src_dir)) if os.path.exists(_method_registry_path(src_dir)) else {}
    change_map = dict(registry.get("method_can_change_argmax", {}))
    src_change = src_registry.get("method_can_change_argmax", {})
    change_map[method_name] = bool(src_change.get(SRC_METHOD_NAME, True))

    stage_map = dict(registry.get("method_stage", {}))
    src_stage = src_registry.get("method_stage", {})
    stage_map[method_name] = src_stage.get(SRC_METHOD_NAME, "stage_kcal")

    registry["ordered_methods"] = ordered
    registry["method_paths"] = method_paths
    registry["method_can_change_argmax"] = change_map
    registry["method_stage"] = stage_map

    _atomic_write_json(dst_registry_path, registry)
    print(f"  Updated method registry at {dst_registry_path}")

    # Rebuild summary by importing the run script's helper
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from run_unified_benchmark import _write_incremental_summary
    json_path, csv_path, methods = _write_incremental_summary(dst_dir)
    print(f"  Rebuilt summary with {len(methods)} methods → {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", required=True)
    parser.add_argument("--dst_dir", required=True)
    args = parser.parse_args()
    merge(args.src_dir, args.dst_dir)


if __name__ == "__main__":
    main()
