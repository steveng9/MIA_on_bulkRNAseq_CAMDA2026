"""Joint attack on PE text / image runs, through evo-mia's own drivers.

Runs a modality's `attacks.run_nfree_mia` (from a deploy copy of evo-mia's committed code:
artifacts/evo_deploy/<modality>/attacks = attacks/<modality> + attacks/common) unchanged --
same audit set, embedder, Ref and transcript -- but replaces its final scoring call so that,
on exactly those inputs, it scores the joint attack (scripts/pe_lira/joint_sparse.py) next
to evo-mia's per-record variants.  Only the 'counts' view (released noised counts) enters the
joint attack; variants needing the true |D| are skipped.

Pass --out_json and --emb_cache_dir pointing inside this repository so nothing is written to
the PE repositories.

    python scripts/pe_lira/joint_modality.py --deploy artifacts/evo_deploy/text \
        --repo ~/privacy/aug-pe-baseline -- --result_folder <run> --dataset openreview ... \
        --out_json results/pe_lira/modalities/openreview_s1.57.json --emb_cache_dir <mine>
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    argv = sys.argv[1:]
    cut = argv.index("--")
    own, rest = argv[:cut], argv[cut + 1:]
    opt = dict(zip(own[::2], own[1::2]))
    deploy = os.path.abspath(os.path.expanduser(opt["--deploy"]))
    repo = os.path.abspath(os.path.expanduser(opt["--repo"]))
    out = rest[rest.index("--out_json") + 1]
    out = os.path.abspath(out)
    rest[rest.index("--out_json") + 1] = out
    if "--emb_cache_dir" in rest:
        i = rest.index("--emb_cache_dir") + 1
        rest[i] = os.path.abspath(rest[i])
    sys.path[:0] = [deploy, repo]          # the deploy copy's `attacks` shadows the repo's own
    os.chdir(repo)                         # the drivers read data paths relative to the repo
    import attacks.run_nfree_mia as drv
    from attacks.common.evaluation import metrics
    from attacks.common.scorers import Context, adapt_variants, score_variant
    from attacks.common.transcript import make_view
    sys.path.insert(len(sys.path), HERE)
    from joint_sparse import joint_scores

    def hook(tr, X, classes, labels, ref_by_class, names=None, alpha=0.05, soft_tau=0.02):
        t0 = time.time()
        ctx = Context(tr, X, classes, ref_by_class, alpha=alpha, soft_tau=soft_tau)
        view = make_view(tr, "counts")
        assert all(rd.get("clean") is None for rs in view["rounds"].values() for rd in rs)
        sigma = float(tr["meta"]["sigma"])
        js, fits = joint_scores(view, ctx, np.asarray(classes), sigma)
        scores = {"joint": js}
        specs = adapt_variants(tr, None)            # as evo-mia's driver: variants adapted to the run's round modes
        for nm in ("A4", "A6", "B5", "C1"):
            if nm in specs and specs[nm]["n"] != "true":
                scores[nm] = score_variant(ctx, nm, spec=specs[nm])["score"]
        report = {"variants": {nm: metrics(labels, s) for nm, s in scores.items()},
                  "joint_fits": [dict(cls=c, N=n, n_hat=nh, s0=s0, cg_info=inf) for c, n, nh, s0, inf in fits],
                  "seconds": time.time() - t0, "sigma": sigma}
        lab = np.asarray(labels)
        print(f"\nsigma {sigma:.2f}; pool {len(lab)} targets ({int(lab.sum())} members); "
              f"fitted member rate {np.mean([f[3] for f in fits]):.2f} (true {lab.mean():.2f}, evaluation only)")
        for nm, m in report["variants"].items():
            t1, t01 = m.get("tpr@0.01", float("nan")), m.get("tpr@0.001", float("nan"))
            print(f"  {nm:6s} AUC {m['auc']:.3f}  TPR@1% {t1:.3f}  TPR@0.1% {t01:.3f}")
        return report, scores

    drv.score_and_report = hook
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sys.argv = ["run_nfree_mia"] + rest
    drv.main()


if __name__ == "__main__":
    main()
