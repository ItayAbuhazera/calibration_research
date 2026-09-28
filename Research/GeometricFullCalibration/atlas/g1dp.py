"""G1-DP — Decoder Panel accessibility audit of G1 (docs/g1_dp_spec.md). TARGET-POOLED HPO — ACCESSIBILITY DIAGNOSTIC ONLY.

Preserves the original G1 fitting unit (docs/g1_conditional_access_spec.md sec. 3): one target-supervised readout per
(checkpoint, arm, outer fold) on T-8k x 1 rows (every outer-train image once, at its preassigned corrupted cell, pooled over the 12 cells);
model selection on the Stage-0 image-grouped inner split; evaluation on the outer-test images under all 13 conditions. Every decoder is
anchored at the native logits z (logits = z + g(x)), as the G1 readout was.

    python -m atlas.g1dp bundle --seed 2 --fold 0                       # stage: data bundle (after extraction)
    python -m atlas.g1dp fit --family lgbm --seed 2 --fold 0 --arms A,B  # stage: one decoder family x (seed, fold) x arms
"""
import argparse
import json
import os
import shutil
import time

import numpy as np

from . import common, decoder_panel as dp, spec, stage0_data, stage0_folds

EXPERIMENT = "G1-DP"
BUNDLES = "results/g1dp/bundles"
FITS = "results/g1dp/fits"
HL_ROOT = "results/g1/hL"
H3_ROOT = "results/g1dp/h322"
PERM_SEED = 20260923                     # = G1 D-shuf permutation seed; partition offsets fit +10, val +20, full +30, test +40
DIMS = {"z": 100, "p": 100, "p42": 100, "zo": 100, "h": 2048, "h3": 1024}
SHUF_BLOCKS = ("p", "h", "h3")
ARMS = {"A": ("z",), "B": ("z", "p"), "C": ("z", "h"), "D": ("z", "h", "p"), "E": ("z", "h", "p42"), "F": ("z", "h", "zo"),
        "H": ("z", "h3"), "I": ("z", "h", "h3"),
        "Cs": ("z", "h~"), "Hs": ("z", "h3~"), "Ds": ("z", "h", "p~")}
ARM_ORDER = ("A", "B", "C", "D", "E", "F", "H", "I", "Cs", "Hs", "Ds")
SEEDS = (2, 4)
FOLDS = range(5)


# ---- bundle ------------------------------------------------------------------------------------------------------------------------
def _perms(fit_ids, val_ids, train, test):
    rng = lambda off: np.random.default_rng(PERM_SEED + off)  # noqa: E731
    return {k: ids[rng(off).permutation(len(ids))] for k, ids, off in (("fit", fit_ids, 10), ("val", val_ids, 20), ("full", train, 30), ("test", test, 40))}


def build_bundle(seed: int, fold: int):
    """All rows one (seed, fold) task needs, block by block (float32 .npy) + manifest. Shuffled blocks follow the G1 D-shuf rule exactly:
    image identity permuted within each partition (inner-fit, inner-val, outer-train refit, outer-test); the permuted image's value is taken
    at the row's own corrupted cell (test: at each condition)."""
    out = f"{BUNDLES}/seed{seed}/fold{fold}"; os.makedirs(out, exist_ok=True)
    o = stage0_folds.load_plan()["outer"][fold]
    train, test = np.array(o["train_idx"]), np.array(o["test_idx"]); inner = np.array(o["inner_fit_mask"], bool); ca = np.array(o["cell_assignment_local"], np.int64)
    assert not set(train.tolist()) & set(test.tolist())
    fit_ids, val_ids = train[inner], train[~inner]; P = _perms(fit_ids, val_ids, train, test)
    cells = stage0_data.load_all_cells(seed); other = stage0_data.load_all_cells(seed, "xckpt")
    src = {}
    for c in spec.CONDITIONS:
        pil = np.load(f"{stage0_data.LAYER_PILOT_ROOT}/checkpoint_seed{seed}/{c}/per_sample.npz")["raw__probe_logits"]
        assert np.array_equal(other[c]["labels"], cells[c]["labels"])
        src[c] = {"z": cells[c]["z"], "p": cells[c]["p"], "p42": pil[:, 11, :].astype(np.float64), "zo": other[c]["p"], "y": cells[c]["labels"],
                  "h": np.load(f"{HL_ROOT}/seed{seed}/h_{c}.npy", mmap_mode="r"), "h3": np.load(f"{H3_ROOT}/seed{seed}/h_{c}.npy", mmap_mode="r")}
    # training rows (outer-train images at their assigned cell); inner-shuffle uses the fit/val permutations, full-shuffle the refit one
    pos_fit, pos_val = np.flatnonzero(inner), np.flatnonzero(~inner)
    shuf_inner_ids = np.empty_like(train); shuf_inner_ids[pos_fit] = P["fit"]; shuf_inner_ids[pos_val] = P["val"]
    arrays = {"train_ids": train, "inner_fit_mask": inner, "cell_assign": ca, "test_ids": test}
    tr = {k: np.empty((len(train), d), np.float32) for k, d in DIMS.items()}
    trs = {f"{k}~inner": np.empty((len(train), DIMS[k]), np.float32) for k in SHUF_BLOCKS} | {f"{k}~full": np.empty((len(train), DIMS[k]), np.float32) for k in SHUF_BLOCKS}
    y_tr = np.empty(len(train), np.int64)
    for ci, cell in enumerate(spec.CELLS):
        m = ca == ci
        if not m.any():
            continue
        s = src[cell]; y_tr[m] = s["y"][train[m]]
        for k in DIMS:
            tr[k][m] = np.asarray(s[k][train[m]], np.float32)
        for k in SHUF_BLOCKS:
            trs[f"{k}~inner"][m] = np.asarray(s[k][shuf_inner_ids[m]], np.float32)
            trs[f"{k}~full"][m] = np.asarray(s[k][P["full"][m]], np.float32)
    te = {k: np.stack([np.asarray(src[c][k][test], np.float32) for c in spec.CONDITIONS]) for k in DIMS}
    tes = {f"{k}~test": np.stack([np.asarray(src[c][k][P["test"]], np.float32) for c in spec.CONDITIONS]) for k in SHUF_BLOCKS}
    y_te = np.stack([src[c]["y"][test] for c in spec.CONDITIONS])
    for k, v in tr.items():
        arrays[f"train_{k}"] = v
    for k, v in trs.items():
        arrays[f"train_{k}"] = v
    for k, v in te.items():
        arrays[f"test_{k}"] = v
    for k, v in tes.items():
        arrays[f"test_{k}"] = v
    arrays["train_y"] = y_tr; arrays["test_y"] = y_te
    man = {}
    for k, v in arrays.items():
        p = f"{out}/{k}.npy"; np.save(p + ".tmp.npy", v); os.replace(p + ".tmp.npy", p); man[k] = {"sha256": common.file_sha(p), "shape": list(v.shape), "dtype": str(v.dtype)}
    common.atomic_json(f"{out}/manifest.json", {"seed": seed, "fold": fold, "conditions": list(spec.CONDITIONS), "files": man,
                                                "perm_seed": PERM_SEED, "created": time.strftime("%Y-%m-%dT%H:%M:%S")})
    return out


def _needed(arms):
    need = {"train_ids", "inner_fit_mask", "cell_assign", "test_ids", "train_y", "test_y", "train_z", "test_z"}
    for a in arms:
        for b in ARMS[a]:
            if b.endswith("~"):
                k = b[:-1]; need |= {f"train_{k}~inner", f"train_{k}~full", f"test_{k}~test"}
            else:
                need |= {f"train_{b}", f"test_{b}"}
    return sorted(need)


def stage_bundle(seed, fold, arms, scratch):
    """Copy only the blocks these arms need to node-local scratch; verify sha256 against the manifest."""
    src = f"{BUNDLES}/seed{seed}/fold{fold}"; man = json.load(open(f"{src}/manifest.json"))["files"]
    os.makedirs(scratch, exist_ok=True); t0 = time.time(); out = {}
    for k in _needed(arms):
        dst = f"{scratch}/{k}.npy"; shutil.copy(f"{src}/{k}.npy", dst)
        assert common.file_sha(dst) == man[k]["sha256"], f"bundle checksum mismatch: {k}"
        out[k] = np.load(dst, mmap_mode="r")
    return out, time.time() - t0


# ---- arm matrices -------------------------------------------------------------------------------------------------------------------
def train_matrix(B, arm, part):
    """part: 'inner' (rows used for model selection; shuffled blocks use the fit/val permutations) or 'full' (refit rows)."""
    cols = []
    for b in ARMS[arm]:
        cols.append(np.asarray(B[f"train_{b[:-1]}~{part}"] if b.endswith("~") else B[f"train_{b}"], np.float64))
    return np.concatenate(cols, 1)


def test_matrix(B, arm, ci):
    return np.concatenate([np.asarray(B[f"test_{b[:-1]}~test"][ci] if b.endswith("~") else B[f"test_{b}"][ci], np.float64) for b in ARMS[arm]], 1)


def context(seed, fold, arm, inner):
    fi, vi = np.flatnonzero(inner), np.flatnonzero(~inner)
    return dp.HPOContext(EXPERIMENT, {"seed": seed, "fold": fold, "arm": arm, "regime": "T-8k1"}, "target_pooled", [(fi, vi)], dp.nll_objective)


# ---- outputs / restart ------------------------------------------------------------------------------------------------------------
def out_paths(family, seed, fold, arm):
    d = f"{FITS}/{family}/seed{seed}/fold{fold}"
    return {"dir": d, "npz": f"{d}/{arm}.npz", "study": f"{d}/{arm}.study.json", "done": f"{d}/{arm}.done.json"}


def is_complete(family, seed, fold, arm) -> bool:
    p = out_paths(family, seed, fold, arm)
    if not os.path.exists(p["done"]):
        return False
    try:
        d = json.load(open(p["done"]))
        return all(os.path.exists(p[k]) and common.file_sha(p[k]) == d[f"{k}_sha256"] for k in ("npz", "study"))
    except Exception:
        return False


def fit_arm(B, family, seed, fold, arm, device="cpu"):
    t0 = time.time()
    inner = np.asarray(B["inner_fit_mask"], bool)
    X_sel, X_full = train_matrix(B, arm, "inner"), train_matrix(B, arm, "full")
    z_tr, y_tr = np.asarray(B["train_z"], np.float64), np.asarray(B["train_y"], np.int64)
    ctx = context(seed, fold, arm, inner)
    # real arms: X_sel == X_full. Shuffled controls: selection on the inner-partition permutation, refit on the outer-train
    # permutation (the G1 D-shuf rule).
    model, rec = dp.fit_decoder(family, X_sel, y_tr, "classification", ctx, offset=z_tr, device=device, X_refit=X_full)
    n = len(B["test_ids"]); C = len(spec.CONDITIONS)
    cor = np.zeros((C, n), np.uint8); nll = np.zeros((C, n), np.float32); pred = np.zeros((C, n), np.int16)
    for ci in range(C):
        zt = np.asarray(B["test_z"][ci], np.float64); yt = np.asarray(B["test_y"][ci], np.int64)
        lg = model.predict(test_matrix(B, arm, ci), zt); lg = lg - lg.max(1, keepdims=True); lse = np.log(np.exp(lg).sum(1))
        cor[ci] = lg.argmax(1) == yt; nll[ci] = lse - lg[np.arange(n), yt]; pred[ci] = lg.argmax(1)
    return {"test_idx": np.asarray(B["test_ids"]), "conditions": np.array(spec.CONDITIONS), "seed": seed, "fold": fold, "arm": arm, "family": family,
            "keys": np.array(ARMS[arm]), "correct": cor, "nll": nll, "pred": pred, "wall_time_s": time.time() - t0}, rec


def run(family, seed, fold, arms, device="cpu", force=False):
    todo = [a for a in arms if force or not is_complete(family, seed, fold, a)]
    print(f"{family} seed{seed} fold{fold}: todo {todo}; skip {[a for a in arms if a not in todo]}", flush=True)
    if not todo:
        return
    scratch = f"/tmp/dp_{os.environ.get('SLURM_JOB_ID', 'local')}_{os.environ.get('SLURM_ARRAY_TASK_ID', '0')}_{family}_{seed}_{fold}"
    try:
        B, copy_s = stage_bundle(seed, fold, todo, scratch)
        for arm in todo:
            rec_npz, rec = fit_arm(B, family, seed, fold, arm, device)
            rec["scratch_copy_seconds"] = copy_s; rec["host"] = os.uname().nodename; rec["cpus"] = dp.n_threads()
            p = out_paths(family, seed, fold, arm); os.makedirs(p["dir"], exist_ok=True)
            np.savez(p["npz"] + ".tmp.npz", **rec_npz); os.replace(p["npz"] + ".tmp.npz", p["npz"])
            dp.save_record(p["study"], rec)
            common.atomic_json(p["done"], {"npz_sha256": common.file_sha(p["npz"]), "study_sha256": common.file_sha(p["study"]),
                                           "finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "wall_time_s": rec_npz["wall_time_s"]})
            print(f"  {arm}: selected {rec['selected']} ({rec_npz['wall_time_s']:.0f}s)", flush=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


# ---- array mapping ----------------------------------------------------------------------------------------------------------------
def units(split_arms: bool):
    """Deterministic array-index mapping: (seed, fold[, arm]) in lexicographic order."""
    if split_arms:
        return [(s, f, a) for s in SEEDS for f in FOLDS for a in ARM_ORDER]
    return [(s, f, None) for s in SEEDS for f in FOLDS]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=("bundle", "fit", "fit_index"))
    ap.add_argument("--seed", type=int, choices=SEEDS); ap.add_argument("--fold", type=int, choices=FOLDS)
    ap.add_argument("--family", choices=dp.FAMILIES); ap.add_argument("--arms", default=",".join(ARM_ORDER))
    ap.add_argument("--index", type=int); ap.add_argument("--split_arms", action="store_true")
    ap.add_argument("--device", default="cpu"); ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.stage == "bundle":
        print(build_bundle(a.seed, a.fold))
    elif a.stage == "fit":
        run(a.family, a.seed, a.fold, a.arms.split(","), a.device, a.force)
    else:
        s, f, arm = units(a.split_arms)[a.index]
        run(a.family, s, f, [arm] if arm else list(ARM_ORDER), a.device, a.force)
