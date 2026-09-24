"""Loader for regime-map artifacts in the Stage 0 format (z, p, labels, n; float64 arrays)."""
import numpy as np

from . import spec

ROOT = "results/regime_map"


def load_cell(state: str, cell: str, root: str = ROOT):
    d = np.load(f"{root}/{state}/{cell}.npz")
    z, p, y = d["z"].astype(np.float64), d["p"].astype(np.float64), d["labels"].astype(np.int64)
    assert z.shape == p.shape and z.shape[1] == spec.NUM_CLASSES and y.shape == (z.shape[0],)
    return {"z": z, "p": p, "labels": y, "base_pred": z.argmax(1), "n": z.shape[0]}


def load_all_cells(state: str, root: str = ROOT):
    return {c: load_cell(state, c, root) for c in spec.CONDITIONS}
