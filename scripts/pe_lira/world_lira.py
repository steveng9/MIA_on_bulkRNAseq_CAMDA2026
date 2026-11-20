"""Per-record LiRA for PE-Tabular, using the other testbed worlds as shadow runs.

A world is a full, independent PE run (its own private set, seed and transcript).
Records recur across the 16 worlds of a setting, as members in some and non-members
in others, so for a target record x in world w the other worlds already hold x's
attack score "in" and "out" -- what a LiRA shadow stack provides.  Calibration is
leave-one-world-out: world w is never used for its own references.

Per variant (each world's scores z-scored over its targets first, label-free):
  raw      the variant's score
  centred  s - mean of x's scores in the other worlds (per-record z, MeLoMIA-like)
  out_z    (s - mu_out(x)) / sd_out                    (offline LiRA)
  lira     log N(s; mu_in(x), sd_in) - log N(s; mu_out(x), sd_out), spreads pooled
           over records (Carlini et al. 2022, few-shadow regime)
A record with no "in" (or "out") score in the other worlds falls back to the pooled
in (out) mean of those worlds.  Metrics pooled over worlds, as in evo-mia's
run_worlds_eval; delta CIs from a paired bootstrap over worlds.

    python scripts/pe_lira/world_lira.py results/pe_lira/breast-cancer_eps10
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve

# Only variants that read the released (noised) counts / selections and estimate |D| from the
# release.  A2 and A2w take the true class sizes from the clean histogram, so they are excluded.
MAIN = ["A3", "A4", "A5", "A6", "B5", "C1", "A4d", "A4n"]


def tpr(y, s, f):
    a, b, _ = roc_curve(y, s)
    return float(np.interp(f, a, b))


def metrics(y, s):
    return roc_auc_score(y, s), tpr(y, s, 0.01), tpr(y, s, 0.001)


def calibrate(W, v):
    """W: list of world dicts. Returns per-world dict of calibrated score arrays."""
    Z = [(w[v] - w[v].mean()) / (w[v].std() + 1e-12) for w in W]
    out = []
    for i, w in enumerate(W):
        ins, outs = {}, {}
        all_in, all_out = [], []
        for j, u in enumerate(W):
            if j == i:
                continue
            for h, s, m in zip(u["row_hash"], Z[j], u["labels"]):
                (ins if m else outs).setdefault(h, []).append(s)
                (all_in if m else all_out).append(s)
        g_in, g_out = np.mean(all_in), np.mean(all_out)
        # pooled spreads: deviations of each record's scores from its own mean
        d_in = [x - np.mean(L) for L in ins.values() if len(L) > 1 for x in L]
        d_out = [x - np.mean(L) for L in outs.values() if len(L) > 1 for x in L]
        sd_in, sd_out = np.std(d_in), np.std(d_out)
        s = Z[i]
        mu_in = np.array([np.mean(ins[h]) if h in ins else g_in for h in w["row_hash"]])
        mu_out = np.array([np.mean(outs[h]) if h in outs else g_out for h in w["row_hash"]])
        mu_all = np.array([np.mean(ins.get(h, []) + outs.get(h, [])) if (h in ins or h in outs)
                           else 0.0 for h in w["row_hash"]])
        miss = np.mean([(h not in ins) or (h not in outs) for h in w["row_hash"]])
        out.append({"raw": s, "centred": s - mu_all, "out_z": (s - mu_out) / sd_out,
                    "lira": norm.logpdf(s, mu_in, sd_in) - norm.logpdf(s, mu_out, sd_out),
                    "_miss": miss})
    return out


def main():
    d = sys.argv[1]
    files = sorted(glob.glob(f"{d}/*_w*.npz"), key=lambda p: int(p.rsplit("_w", 1)[1][:-4]))
    W = [dict(np.load(f, allow_pickle=True)) for f in files]
    y = np.concatenate([w["labels"] for w in W])
    single = np.concatenate([w["copies"] <= 1 for w in W])
    wid = np.concatenate([np.full(len(w["labels"]), i) for i, w in enumerate(W)])
    print(f"{os.path.basename(d)}: {len(W)} worlds, {len(y)} targets pooled "
          f"({int(y.sum())} members; {int((~single).sum())} multi-copy member targets)")
    rng = np.random.default_rng(0)
    boots = [rng.integers(0, len(W), len(W)) for _ in range(500)]
    summary = {}
    for v in [v for v in MAIN if v in W[0]]:
        raw_unstd = np.concatenate([w[v] for w in W])
        C = calibrate(W, v)
        res = {"raw_unstandardised": metrics(y, raw_unstd)}
        for k in ("raw", "centred", "out_z", "lira"):
            s = np.concatenate([c[k] for c in C])
            res[k] = metrics(y, s)
            res[k + "_single"] = metrics(y[single], s[single])
        # paired world bootstrap of lira - raw
        sr, sl = np.concatenate([c["raw"] for c in C]), np.concatenate([c["lira"] for c in C])
        dA, dT = [], []
        for b in boots:
            idx = np.concatenate([np.flatnonzero(wid == i) for i in b])
            dA.append(roc_auc_score(y[idx], sl[idx]) - roc_auc_score(y[idx], sr[idx]))
            dT.append(tpr(y[idx], sl[idx], 0.01) - tpr(y[idx], sr[idx], 0.01))
        res["delta_auc_ci"] = list(np.percentile(dA, [2.5, 97.5]))
        res["delta_tpr01_ci"] = list(np.percentile(dT, [2.5, 97.5]))
        res["missing_ref_frac"] = float(np.mean([c["_miss"] for c in C]))
        summary[v] = res
        f = lambda t: f"{t[0]:.3f} / {t[1]:.3f} / {t[2]:.3f}"
        print(f"  {v:4s} raw(unstd) {f(res['raw_unstandardised'])} | raw {f(res['raw'])} | centred {f(res['centred'])} | "
              f"out_z {f(res['out_z'])} | LiRA {f(res['lira'])}   [AUC / TPR@1% / TPR@0.1%]")
        print(f"       LiRA - raw: dAUC CI {np.round(res['delta_auc_ci'], 3)}, dTPR@1% CI {np.round(res['delta_tpr01_ci'], 3)}; "
              f"single-copy LiRA {f(res['lira_single'])} vs raw {f(res['raw_single'])}; "
              f"records lacking in or out refs {res['missing_ref_frac']:.1%}")
    json.dump(summary, open(f"{d}/lira_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
