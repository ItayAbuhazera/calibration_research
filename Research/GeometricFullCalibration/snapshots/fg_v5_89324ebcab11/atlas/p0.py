"""Stage P0 (CPU): shared caches, role manifests, subset, duplicate audit.  python -m atlas.p0"""
import hashlib, json, os, sys, time
import numpy as np
from . import data, spec


def rowhash(a):
    return [hashlib.blake2b(np.ascontiguousarray(a[i]).tobytes(), digest_size=16).digest() for i in range(a.shape[0])]


def main():
    os.makedirs(data.SHARED, exist_ok=True)
    out = {"stage": "p0", "conditions": list(spec.CONDITIONS)}
    # shared corrupted test sets (independent of the checkpoint), corrected protocol
    p = data.full_sets_path()
    a2 = data.load_seed_arrays(2)
    test_x, test_y = np.asarray(a2["test"][0]), a2["test"][1]
    a4 = data.load_seed_arrays(4)
    assert np.array_equal(a4["test"][1], test_y) and np.array_equal(np.asarray(a4["test"][0]), test_x)
    np.save(f"{data.SHARED}/test_labels.npy", test_y)
    if not os.path.exists(p):
        from utils.calibration_utils import get_all_data_as_numpy, load_cifar_c_loader
        from utils import preprocessing_protocol as pp
        tf = pp.eval_transform("cifar100", pp.PROTOCOL_CORRECTED, corruption=True)
        full = np.lib.format.open_memmap(p + ".tmp.npy", mode="w+", dtype=np.float32, shape=(13 * 10000, 3, 32, 32))
        full[data.cond_slice("clean")] = test_x
        for cell in spec.CELLS:
            c, s = cell.rsplit("_s", 1)
            loader = load_cifar_c_loader("cifar100", c, int(s), tf, 250, cifar_c_dir=data.CIFAR_C)
            x, y = get_all_data_as_numpy(loader)
            assert np.array_equal(y, test_y), cell
            full[data.cond_slice(cell)] = x
        full.flush(); del full
        os.replace(p + ".tmp.npy", p)
    full = data.load_full_sets()
    out["full_sets_sha256_first_1e6"] = spec.sha_arr(np.asarray(full[:1000]).ravel()[:10**6])
    sub = data.make_subset(test_y)
    np.save(f"{data.SHARED}/subset_ids.npy", sub)
    out["subset"] = {"n": int(len(sub)), "sha256": spec.sha_arr(sub), "per_class": spec.SUBSET_PER_CLASS, "seed": spec.SUBSET_SEED,
                     "class_counts_min_max": [int(np.bincount(test_y[sub]).min()), int(np.bincount(test_y[sub]).max())]}
    # hashes for exact-duplicate audit (test images vs each checkpoint's bank and validation rows)
    th = rowhash(test_x)
    for seed in spec.DEV_SEEDS:
        arrs = data.load_seed_arrays(seed)
        os.makedirs(data.seed_dir(seed), exist_ok=True)
        val_y = arrs["val"][1]
        roles = data.make_roles(val_y)
        bank_h = rowhash(arrs["train"][0]); val_h = rowhash(arrs["val"][0])
        bset = {}
        for i, h in enumerate(bank_h):
            bset.setdefault(h, []).append(i)
        dup_val = [(i, len(bset[h])) for i, h in enumerate(val_h) if h in bset]
        dup_test = [(i, len(bset[h])) for i, h in enumerate(th) if h in bset]
        bank_dup_groups = sum(1 for v in bset.values() if len(v) > 1)
        conflict = sum(1 for v in bset.values() if len(v) > 1 and len(set(arrs["train"][1][v])) > 1)
        rec = {"seed": seed, "roles": {k: v.tolist() for k, v in roles.items()},
               "role_sizes": {k: int(len(v)) for k, v in roles.items()},
               "role_hashes": {k: spec.sha_arr(v) for k, v in roles.items()},
               "val_labels_hash": spec.sha_arr(val_y), "train_labels_hash": spec.sha_arr(arrs["train"][1]),
               "role_class_counts_min_max": {k: [int(np.bincount(val_y[v], minlength=100).min()), int(np.bincount(val_y[v], minlength=100).max())] for k, v in roles.items()},
               "duplicate_audit": {"val_rows_identical_to_a_bank_image": len(dup_val), "test_images_identical_to_a_bank_image": len(dup_test),
                                   "bank_duplicate_groups": bank_dup_groups, "bank_duplicate_groups_with_conflicting_labels": conflict,
                                   "note": "duplicates are recorded by content hash only; no unique original index is inferred for them. "
                                           "Queries never appear in the bank by original ID (val and test are disjoint splits of CIFAR-100)."}}
        stage0 = f"results/residual_study/stage0_stage1/manifest_seed{seed}.json"
        if os.path.exists(stage0):
            m = json.load(open(stage0))
            vm = np.asarray(m["val_original_index_in_materialized_order"])
            rec["original_index_of_role_rows"] = {k: vm[v].tolist() for k, v in roles.items()}
            rec["original_index_note"] = "from the residual-study Stage-0 hash mapping; ambiguous for pixel-identical duplicates"
            rec["val_train_disjoint_by_original_index"] = bool(set(vm.tolist()).isdisjoint(set(m["train_original_index_in_materialized_order"])))
        json.dump(rec, open(f"{data.seed_dir(seed)}/roles.json", "w"))
        out[f"seed{seed}"] = {k: v for k, v in rec.items() if k not in ("roles", "original_index_of_role_rows")}
    json.dump(out, open(f"{data.SHARED}/p0_manifest.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float)[:3000])


if __name__ == "__main__":
    main()
