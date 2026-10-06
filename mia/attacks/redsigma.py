"""RedSigma -- the other winning CAMDA-26 red-team attack, run on our grid.

Tucker et al.'s submission (https://github.com/OwenTucker/ELSA_REDSIGMA) is a
shadow-free, distance-to-synthetic-data attack with a different scoring rule
per generator, chosen by hand for the four challenge targets:

    knn200   mean cosine distance to the 200 nearest synthetic rows   (NoisyDiffusion)
    knn500   the same with 500 neighbours                             (DP-PGM)
    cvae     BRCA:     nearest-synthetic cosine distance, genes weighted by the
                       per-gene Gaussian KL between synthetic and reference
                       marginals, minus the same distance to the reference
             COMBINED: drop the top 100 principal components, then nearest-
                       synthetic minus nearest-reference cosine distance
    mvn      log-likelihood under a Gaussian fitted to the synthetic data (MVN)

The scoring functions are THEIR code, imported by file path from
`paths.REDSIGMA_REPO` (their package is also called `mia`, so it cannot be
imported by name).  Only the dispatcher in their `elsa_attack.py` is restated
here, because it reads files rather than arrays; `tests/test_redsigma.py` holds
it to their submitted prediction files, which it reproduces on all eight
challenge datasets.

Three things about the submission are kept as they were, because changing them
would be attacking with something other than what won:

  * Preprocessing.  Candidates, synthetic data and reference are each z-scored
    with their OWN StandardScaler, then all three again with one fitted on the
    candidates.  Scores are therefore transductive: they depend on the whole
    candidate set, which in the challenge and here is the full cohort.
  * Reference.  COMBINED uses the challenge's auxiliary set.  BRCA has none, and
    their code then substitutes the candidate set itself.  For `cvae` on BRCA
    that makes the reference distance a distance-to-self (zero up to rounding),
    so the rule reduces to the KL-weighted nearest-synthetic distance.  For the
    two `knn` rules the reference only shifts and rescales every score by the
    same constants, so it cannot change any rank metric on either cohort.
  * Labels.  The challenge released no synthetic labels, so their `mvn` rule ran
    with one pooled Gaussian.  `use_labels=True` gives it the released labels,
    which is the per-class likelihood their code was written for.

`rule="auto"` picks the rule their dispatcher would for our generator names.
Generators they never saw (tabsyn, tabpfn, ...) fall through to `mvn`, exactly
as an unrecognised name does in their code; pass a rule explicitly to do better.

To run it on a new generator: build the target, add its name to `generators`
in `configs/experiments/redsigma_{brca,combined}.yaml`, and re-run

    python scripts/run_experiment.py configs/experiments/redsigma_brca.yaml

(seconds per target, CPU only).  If the generator belongs to one of their four
families, add it to FAMILY_RULE so `auto` gives it that family's rule.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .. import datasets as D
from .. import paths
from .. import targets as T
from .base import Attack, register

RULES = ("knn200", "knn500", "cvae", "mvn")

# Our generator names -> the rule their dispatcher applies to that family.
FAMILY_RULE = {
    "nd": "knn200",
    "pgm": "knn500", "pgg": "knn500", "dpsynth": "knn500",
    "cvae": "cvae", "dpcvae": "cvae",
    "mvn": "mvn",
}


@lru_cache(maxsize=1)
def _theirs():
    """Their `mia_variants.py` and `attack_utils.py`, loaded by file path."""
    mods = []
    for stem in ("mia_variants", "attack_utils"):
        f = paths.REDSIGMA_REPO / "src" / "mia" / "models" / f"{stem}.py"
        if not f.exists():
            raise FileNotFoundError(
                f"{f} not found. Clone https://github.com/OwenTucker/ELSA_REDSIGMA "
                "and point CAMDA_REDSIGMA_REPO at it (see mia/paths.py).")
        spec = importlib.util.spec_from_file_location(f"redsigma_{stem}", f)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mods.append(mod)
    return tuple(mods)


def rule_for(generator: str) -> str:
    base = generator.partition("@")[0]
    return FAMILY_RULE.get(base, "mvn")     # their dispatcher's else-branch


def redsigma_scores(X_test, X_syn, X_ref=None, rule: str = "knn200",
                    combined: bool = False, y_syn=None) -> np.ndarray:
    """One RedSigma rule on raw expression arrays; higher = more likely member.

    `combined` selects which of their two CVAE rules runs (they switch on the
    cohort name).  `y_syn` is used by `mvn` only; None is one pooled class.
    """
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}; known: {RULES}")
    V, U = _theirs()
    std = lambda Z: StandardScaler().fit_transform(np.asarray(Z, dtype=np.float64))  # noqa: E731

    # MIADataLoader: each set on its own scaler ...
    test_l, syn_l = std(X_test), std(X_syn)
    # ... then ELSARedTeamAttack.run_attack: one more, fitted on the candidates.
    scaler = StandardScaler()
    test = scaler.fit_transform(test_l)
    syn = scaler.transform(syn_l)
    ref = scaler.transform(std(X_ref)) if X_ref is not None else test

    if rule in ("knn200", "knn500"):
        k = min(int(rule[3:]), syn.shape[0] - 1)
        return V.attack_gan_leaks_lira_cosine(test, syn, ref, k=k)

    if rule == "cvae" and combined:
        n_comp = min(syn.shape[0] - 1, ref.shape[0] - 1, syn.shape[1])
        pca = PCA(n_components=n_comp, random_state=42).fit(np.vstack([syn, ref]))
        k = min(100, n_comp)
        res = []
        for Z in (test, syn, ref):
            Zp = pca.transform(Z).copy()
            Zp[:, :k] = 0.0
            res.append(pca.inverse_transform(Zp))
        return V.attack_logan_cosine_pctl(*res, k=1, pctl=75)

    if rule == "cvae":
        w = U.compute_gene_weights_kl(syn, ref)
        d_syn = U.weighted_cosine_min_distances(test, syn, w)
        d_ref = U.weighted_cosine_min_distances(test, ref, w)
        return (d_ref / (np.percentile(d_ref, 25) + 1e-10)
                - d_syn / (np.percentile(d_syn, 25) + 1e-10))

    # mvn: pseudo-label each candidate by its nearest synthetic row, then the
    # per-class Gaussian log-likelihood (on the once-scaled arrays, as theirs).
    y = (np.full(len(syn), "Unknown") if y_syn is None
         else np.asarray(y_syn).astype(str).ravel())
    classes, y_enc = np.unique(y, return_inverse=True)
    pseudo = classes[V._assign_pseudo_labels(test, syn, y_enc)]
    return V._compute_mvn_wb_scores(test_l, pseudo, syn_l, y)["log_likelihood"]


@dataclass
class RedSigma(Attack):
    rule: str = "auto"            # auto | knn200 | knn500 | cvae | mvn
    use_labels: bool = False      # False = as submitted (no labels were released)

    name = "redsigma"

    def tag(self) -> str:
        return self.rule + ("_labels" if self.use_labels else "")

    def score(self, dataset: str, generator: str, split: int) -> np.ndarray:
        rule = rule_for(generator) if self.rule == "auto" else self.rule
        X = D.load_expression(dataset).values
        tg = T.load_target(dataset, generator, split)
        ref = D.load_reference(dataset)
        return redsigma_scores(
            X, tg["X"], None if ref is None else ref.values, rule=rule,
            combined="COMBINED" in dataset.upper(),
            y_syn=tg["y_str"] if self.use_labels else None)


register(RedSigma)
