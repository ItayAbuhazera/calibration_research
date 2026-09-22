"""Refresh results/atlas/ledger.json job states from sacct and write a human summary: python -m atlas.ledger"""
import json, subprocess

def main():
    L = json.load(open("results/atlas/ledger.json"))
    ids = sorted({e["job_id"] for e in L})
    out = subprocess.check_output(["sacct", "-X", "-n", "-P", "-j", ",".join(ids), "--format=JobID,JobName,State,Elapsed,ExitCode,Start,End,NodeList"], text=True)
    rows = [r.split("|") for r in out.strip().splitlines()]
    by = {}
    for r in rows:
        by.setdefault(r[0].split("_")[0], []).append(r)
    lines = ["| stage | job | tasks (state counts) | elapsed sample | deps |", "|---|---|---|---|---|"]
    for e in L:
        rs = by.get(e["job_id"], [])
        c = {}
        for r in rs:
            c[r[2].split()[0]] = c.get(r[2].split()[0], 0) + 1
        e["state_counts"] = c
        lines.append(f"| {e['stage']} | {e['job_id']} | {c} | {rs[0][3] if rs else ''} | {','.join(e['deps'])} |")
    json.dump(L, open("results/atlas/ledger.json", "w"), indent=1)
    open("results/atlas/ledger_status.md", "w").write("\n".join(lines))
    print("\n".join(lines))

if __name__ == "__main__":
    main()
