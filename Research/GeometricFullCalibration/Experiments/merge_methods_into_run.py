"""
Copy method entries from a source benchmark run into an existing destination run,
then rebuild summary_metrics.json/.csv.

By default copies every method in the source that is absent from the destination.
Pass --methods to restrict which names are copied.

Usage:
    python Experiments/merge_methods_into_run.py \
        --src_dir results/.../seed22/decision_run_v1_factorial \
        --dst_dir results/.../seed22/decision_run_v1 \
        [--methods kcal_factorial_penultimate_replace kcal_factorial_rgcl_blend_frozen ...]
        [--overwrite]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile


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


def merge(src_dir: str, dst_dir: str, methods: list[str] | None, overwrite: bool) -> None:
    src_reg_path = _method_registry_path(src_dir)
    dst_reg_path = _method_registry_path(dst_dir)

    if not os.path.exists(src_reg_path):
        sys.exit(f"ERROR: source registry not found: {src_reg_path}")
    if not os.path.exists(dst_reg_path):
        sys.exit(f"ERROR: destination registry not found: {dst_reg_path}")

    src_registry = _load_json(src_reg_path)
    dst_registry = _load_json(dst_reg_path)

    src_methods = src_registry.get("ordered_methods", [])
    dst_methods_set = set(dst_registry.get("ordered_methods", []))

    # Determine which methods to copy
    if methods:
        to_copy = [m for m in methods if m in src_methods]
        missing = [m for m in methods if m not in src_methods]
        if missing:
            print(f"  WARNING: requested methods not found in source: {missing}")
    else:
        # Copy everything from source not already in destination
        to_copy = [m for m in src_methods if m not in dst_methods_set or overwrite]

    if not to_copy:
        print("  Nothing to copy (all methods already present; use --overwrite to replace).")
        return

    print(f"  Copying {len(to_copy)} method(s): {to_copy}")

    src_store = _method_store_dir(src_dir)
    dst_store = _method_store_dir(dst_dir)

    dst_ordered = list(dst_registry.get("ordered_methods", []))
    dst_paths = dict(dst_registry.get("method_paths", {}))
    dst_change = dict(dst_registry.get("method_can_change_argmax", {}))
    dst_stage = dict(dst_registry.get("method_stage", {}))

    src_paths = src_registry.get("method_paths", {})
    src_change = src_registry.get("method_can_change_argmax", {})
    src_stage = src_registry.get("method_stage", {})

    copied = []
    for method_name in to_copy:
        src_method_dir = os.path.join(src_store, method_name)
        dst_method_dir = os.path.join(dst_store, method_name)

        # Validate source files
        missing_files = [f for f in ("entry.json", "probs.npy", "meta.json")
                         if not os.path.exists(os.path.join(src_method_dir, f))]
        if missing_files:
            print(f"  SKIP {method_name}: source missing {missing_files}")
            continue

        os.makedirs(dst_method_dir, exist_ok=True)
        for fname in ("entry.json", "probs.npy", "meta.json"):
            shutil.copy2(
                os.path.join(src_method_dir, fname),
                os.path.join(dst_method_dir, fname),
            )

        if method_name not in dst_ordered:
            dst_ordered.append(method_name)
        dst_paths[method_name] = {
            "entry": os.path.join(dst_method_dir, "entry.json"),
            "probs": os.path.join(dst_method_dir, "probs.npy"),
            "meta":  os.path.join(dst_method_dir, "meta.json"),
        }
        dst_change[method_name] = bool(src_change.get(method_name, True))
        dst_stage[method_name] = src_stage.get(method_name, "stage4_kcal_factorial")
        copied.append(method_name)

    if not copied:
        print("  No methods were successfully copied.")
        return

    dst_registry["ordered_methods"] = dst_ordered
    dst_registry["method_paths"] = dst_paths
    dst_registry["method_can_change_argmax"] = dst_change
    dst_registry["method_stage"] = dst_stage
    _atomic_write_json(dst_reg_path, dst_registry)
    print(f"  Updated registry at {dst_reg_path}")

    # Rebuild summary
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from run_unified_benchmark import _write_incremental_summary
    _, csv_path, methods_all = _write_incremental_summary(dst_dir)
    print(f"  Rebuilt summary: {len(methods_all)} methods → {csv_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", required=True)
    ap.add_argument("--dst_dir", required=True)
    ap.add_argument("--methods", nargs="*", default=None,
                    help="Specific method names to copy (default: all new methods)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Replace methods already present in destination")
    args = ap.parse_args()
    merge(args.src_dir, args.dst_dir, args.methods, args.overwrite)


if __name__ == "__main__":
    main()
