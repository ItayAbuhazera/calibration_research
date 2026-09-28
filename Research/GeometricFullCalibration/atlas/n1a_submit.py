"""Submit N1a (docs/n1a_action_ambiguity_spec.md) from an immutable snapshot: CPU arrays over folds per (base, family, arm), then aggregation.

    python -m atlas.n1a_submit snapshots/<dir>
"""
import sys

from . import n1a
from .g1_submit import PY, sb as _sb
from . import g1_submit

g1_submit.LEDGER = "results/n1a/ledger.json"


def main(snap):
    fits = []
    for base in n1a.BASES:
        for fam in n1a.FAMILIES:
            for arm in ("Z", "ZZo"):
                fits.append(_sb(snap, f"n1a_b{base}_{fam}_{arm}", f"{PY} -m atlas.n1a_fit --base {base} --family {fam} --fold $SLURM_ARRAY_TASK_ID --arm {arm}",
                                array="0-4", hours="02:00:00", mem="16G"))
    _sb(snap, "n1a_aggregate", f"{PY} -m atlas.n1a_aggregate", deps=fits, hours="02:00:00", mem="48G")


if __name__ == "__main__":
    main(sys.argv[1])
