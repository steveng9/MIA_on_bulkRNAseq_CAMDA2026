"""Give the corrected DP-PGM targets whose checkpoint is slim their fitted model.

A white-box attack on a DP-PGM target reads the fitted graphical model
(`gen._joint_fitter._model`).  Targets built before 2026-09-25 kept it; those
built after were saved slim (PGMGenerator.save drops the model by default, to
save disk), so they cannot be sampled or read.  On the paper's corrected DP-PGM
that is splits 4 and 5 of both TCGA cohorts.

The fit is a deterministic function of the training split and the recorded
seed (`fit` seeds numpy's global generator, and every later draw follows from
it), so the model can be rebuilt: refit with the recipe and seed in meta.json,
then check that it is *the* target before saving:

  release   in the refitting process, sampling n rows right after the fit
            re-emits the released data and labels (rows to 1e-5 relative,
            labels exactly);
  state     the bin edges and the measured tables equal the slim checkpoint's;
  saved     the checkpoint written and read back has the refit's potentials,
            clique by clique, exactly.

The rows can be re-emitted only in the refitting process: the upstream sampler
draws from numpy's global generator, whose state after a fit is not saved, so a
model loaded later samples the same distribution but different rows (true of
the checkpoints that kept their model at build time too).  `saved` is the
check that survives a reload.

A model that fails a check is reported and not saved.  The slim checkpoint is
kept beside the full one as generator.slim.pt.  Run again on a target that
already has generator.slim.pt to re-verify the saved model against a new refit.

    python scripts/whitebox/materialise_pgm_models.py --datasets BRCA COMBINED --splits 4 5
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mia import datasets as D  # noqa: E402
from mia import generators as G  # noqa: E402
from mia import targets as T  # noqa: E402

PGM = "pgm@binning=dp_quantile,edge_estimator=threshold,n_bins=16"
ROW_TOL = 1e-5


def _model(gen):
    fitter = getattr(gen._gen, "_joint_fitter", None)
    return getattr(fitter, "_model", None) if fitter is not None else None


def _state(gen) -> dict:
    """What a slim checkpoint keeps that pins the fit: bin edges and tables."""
    g = gen._gen
    return {"edges": [np.asarray(e) for e in g._discretizer._edges],
            "cliques": {k: [tuple(c) for c in v] for k, v in g._marginals.items()}}


def _same_state(a: dict, b: dict) -> str | None:
    if len(a["edges"]) != len(b["edges"]) or not all(
            np.array_equal(x, y) for x, y in zip(a["edges"], b["edges"])):
        return "bin edges differ"
    if a["cliques"] != b["cliques"]:
        return "measured tables differ"
    return None


def _same_potentials(a, b) -> str | None:
    pa, pb = a.potentials, b.potentials
    if list(pa.keys()) != list(pb.keys()):
        return "cliques differ"
    for c in pa.keys():
        if not np.array_equal(np.asarray(pa[c].values), np.asarray(pb[c].values)):
            return f"potential of {c} differs"
    return None


def materialise(dataset: str, split: int) -> str:
    f = T._files(dataset, PGM, split)
    slim_file = f["dir"] / "generator.slim.pt"
    if _model(T.load_model(dataset, PGM, split, device="cpu")) is not None and not slim_file.exists():
        return "kept its model at build time"

    meta = json.loads(f["meta"].read_text())
    base, _ = T.split_name(PGM)
    gen = G.build(base, seed=meta["seed"], device="cpu", verbose=False, **meta["params"])
    X, y, _ = D.training_subset(dataset, split)
    gen.fit(X, y, D.n_classes(dataset))

    tg = T.load_target(dataset, PGM, split)
    X_re, y_re = gen.sample(len(tg["X"]))
    if not np.array_equal(y_re, tg["y_int"]):
        return "FAILED: labels differ from the release - not saved"
    rel = float(np.abs(X_re - tg["X"]).max() / max(np.abs(tg["X"]).max(), 1e-12))
    if rel > ROW_TOL:
        return f"FAILED: rows differ from the release (max relative {rel:.3g}) - not saved"

    slim = G.build(base, seed=meta["seed"], device="cpu", verbose=False,
                   **meta["params"]).load(slim_file if slim_file.exists() else f["model"])
    why = _same_state(_state(gen), _state(slim))
    if why:
        return f"FAILED: {why} from the slim checkpoint - not saved"

    if slim_file.exists():
        done = "re-verified the saved model"
    else:
        shutil.copy2(f["model"], slim_file)
        gen.save(f["model"], keep_model=True)
        done = "saved"
    back = _model(T.load_model(dataset, PGM, split, device="cpu"))
    if back is None:
        return "FAILED: the checkpoint reloads without a model"
    why = _same_potentials(_model(gen), back)
    if why:
        return f"FAILED: the checkpoint is not the refit ({why})"
    return f"{done} (release re-emitted, max relative row difference {rel:.1e}; potentials identical)"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--datasets", nargs="+", default=["BRCA", "COMBINED"])
    p.add_argument("--splits", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    a = p.parse_args()
    bad = 0
    for ds in a.datasets:
        for s in a.splits:
            msg = materialise(ds, s)
            bad += msg.startswith("FAILED")
            print(f"  {ds} split {s}: {msg}", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
