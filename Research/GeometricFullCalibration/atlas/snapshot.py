"""Immutable code snapshot for submitted jobs: python -m atlas.snapshot [tag]

Copies the code (atlas, Calibrators, Net, utils, data/*.py, tests) to snapshots/<tag>_<treehash>/, symlinks ``results``,
``logs`` and the dataset directories back to the shared tree, hashes every copied file and marks the tree read-only.
Jobs run with cwd=snapshot and PYTHONPATH=snapshot, so later edits to the working tree cannot change pending tasks.
"""
import hashlib, json, os, shutil, stat, subprocess, sys, time

SRC = os.getcwd()
DIRS = ["atlas", "Calibrators", "Net", "utils", "tests", "Metrics", "Losses"]


def main(tag="atlas_v1"):
    files = []
    for d in DIRS:
        for r, _, fs in os.walk(d):
            if "__pycache__" in r:
                continue
            files += [os.path.join(r, f) for f in fs if f.endswith((".py", ".txt", ".md"))]
    files += [os.path.join("data", f) for f in os.listdir("data") if f.endswith(".py")]
    files += [f for f in ("docs/atlas_program_spec.md", "docs/fixed_gate_study_spec.md", "docs/stage0_execution_spec.md", "docs/stage0_evidence_ablation_spec.md", "docs/regime_map_pilot_spec.md", "docs/regime_map_pilot_spec_v2.md") if os.path.exists(f)]
    h = hashlib.sha256()
    manifest = {}
    for f in sorted(files):
        b = open(f, "rb").read(); manifest[f] = hashlib.sha256(b).hexdigest(); h.update(f.encode() + b)
    tree = h.hexdigest()[:12]
    dst = os.path.join("snapshots", f"{tag}_{tree}")
    if os.path.exists(dst):
        print("snapshot exists:", dst); return dst
    for f in sorted(files):
        os.makedirs(os.path.dirname(os.path.join(dst, f)), exist_ok=True); shutil.copy2(f, os.path.join(dst, f))
    for d in DIRS:
        for r, _, fs in os.walk(os.path.join(dst, d)):
            if not os.path.exists(os.path.join(r, "__init__.py")) and d in ("Calibrators", "Net", "utils", "atlas"):
                pass
    for link in ("results", "logs"):
        os.symlink(os.path.abspath(link), os.path.join(dst, link))
    for extra in os.listdir("data"):
        p = os.path.join("data", extra)
        if os.path.isdir(p) and not os.path.exists(os.path.join(dst, "data", extra)):
            os.symlink(os.path.abspath(p), os.path.join(dst, "data", extra))
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        diff = hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD", "--binary", "--", "."])).hexdigest()
    except Exception:
        head = diff = None
    json.dump({"tag": tag, "tree_hash": tree, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "git_head": head, "tracked_diff_sha256": diff,
               "note": "includes uncommitted repairs (preprocessing protocol etc.); a HEAD checkout alone would omit them", "files": manifest},
              open(os.path.join(dst, "SNAPSHOT.json"), "w"), indent=1)
    for r, ds, fs in os.walk(dst, followlinks=False):
        for f in fs:
            p = os.path.join(r, f)
            if not os.path.islink(p):
                os.chmod(p, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    print(dst); return dst


if __name__ == "__main__":
    main(*sys.argv[1:])
