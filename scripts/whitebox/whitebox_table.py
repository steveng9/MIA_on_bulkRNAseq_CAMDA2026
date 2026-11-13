"""White box against black box: one row per (attack, generator), both cohorts.

Black-box cells are the paper grid's (same rows as scripts/paper_figures.py);
white-box cells come from configs/experiments/whitebox_*.yaml and, for
MeLoMIA-TabSyn, melomia_tabsyn_*.yaml.  AUC / TPR at 1% FPR, mean over splits;
a cell with fewer than five splits is starred.

    python scripts/whitebox/whitebox_table.py            # markdown, to stdout
    python scripts/whitebox/whitebox_table.py --latex    # the paper's table body
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from paper_figures import PGM, TABSYN, cell, load_runs  # noqa: E402

COHORTS = ["BRCA", "COMBINED"]
MAHALA = "aux_ridge0.0001_shrink0.5"

# (attack label, generator label, generator, black-box choices, white-box arms)
# where a white-box arm is (label, choices).  Choices list the BRCA tag first
# and the COMBINED tag second where K differs; `cell` takes the first present.
ROWS = [
    ("MahalaMIA", "MVN", "mvn", [("mahalamia", MAHALA)],
     [("model moments", [("mahalamia", MAHALA + "_whitebox100000")])]),
    ("MahalaMIA", "CVAE", "cvae", [("mahalamia", MAHALA)],
     [("model moments", [("mahalamia", MAHALA + "_whitebox100000")])]),
    ("MahalaMIA", "TabSyn", TABSYN, [("mahalamia", MAHALA)],
     [("model moments", [("mahalamia", MAHALA + "_whitebox100000")])]),
    ("MahalaMIA", "DP-PGM (corrected)", PGM, [("mahalamia", MAHALA)],
     [("model moments", [("mahalamia", MAHALA + "_whitebox100000")])]),
    ("MeLoMIA-CVAE", "CVAE", "cvae",
     [("melomia_cvae", "k30_n50_prc"), ("melomia_cvae", "k20_n50_prc")],
     [("target model", [("melomia_cvae", "k30_n50_whitebox_prc"),
                        ("melomia_cvae", "k20_n50_whitebox_prc")])]),
    ("MeLoMIA-TabSyn", "TabSyn", TABSYN,
     [("melomia_tabsyn", "k30_n200_prc"), ("melomia_tabsyn", "k20_n200_prc")],
     [("target model", [("melomia_tabsyn", "k30_n200_whitebox_prc"),
                        ("melomia_tabsyn", "k20_n200_whitebox_prc")])]),
    ("MAMA-MIA", "DP-PGM (corrected)", PGM, [("mamamia_v2", "cl-public_ed-auto_cc")],
     [("true tables and bins", [("mamamia_v2", "cl-true_ed-known_cc")]),
      ("model marginals", [("mamamia_v2", "cl-true_ed-known_cc_sy-model")])]),
]


def fmt(c, latex=False) -> str:
    if c is None:
        return "--" if latex else "—"
    s = f"{c['auc']:.2f} / {c['tpr_at_fpr_0.01']:.2f}"
    if c["n_splits"] < 5:
        s += "$^{*}$" if latex else "*"
    return s


def table(df) -> list[dict]:
    out = []
    for attack, glabel, gen, bb, arms in ROWS:
        for i, (arm, choices) in enumerate(arms):
            row = {"attack": attack if i == 0 else "", "generator": glabel if i == 0 else "",
                   "white box": arm}
            for ds in COHORTS:
                row[f"{ds} black"] = cell(df, ds, gen, bb) if i == 0 else None
                row[f"{ds} white"] = cell(df, ds, gen, choices)
            out.append(row)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--latex", action="store_true")
    a = p.parse_args()
    rows = table(load_runs())
    if a.latex:
        for r in rows:
            cells = [r["attack"], r["generator"], r["white box"]]
            for ds in COHORTS:
                cells += ["" if not r["attack"] and r[f"{ds} black"] is None
                          else fmt(r[f"{ds} black"], True), fmt(r[f"{ds} white"], True)]
            print(" & ".join(cells) + r" \\")
        return
    head = ["Attack", "Generator", "White box reads"] + [
        f"{ds} {k}" for ds in COHORTS for k in ("black", "white")]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for r in rows:
        cells = [r["attack"], r["generator"], r["white box"]]
        for ds in COHORTS:
            cells += ["" if not r["attack"] else fmt(r[f"{ds} black"]), fmt(r[f"{ds} white"])]
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
