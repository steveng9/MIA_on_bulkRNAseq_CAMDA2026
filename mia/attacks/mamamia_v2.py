"""MAMA-MIA v2 -- marginal-ratio membership inference that follows the target.

v1 (`mamamia.py`, the CAMDA-2026 extended abstract) scores a fixed set of
tables: every gene's 1-way marginal and every (gene, label) marginal, on bins
of its own choosing.  That was exactly right for last year's DP-PGM, whose
star of tables is fixed by public configuration.  v2 is for the full paper,
where the target may also measure a data-dependent set of gene-gene tables
(`structure=tree` / `tree_label`: a DP spanning tree of (gene, gene[, label])
marginals, MST's recipe on top of the star).

The principle is unchanged from MAMA-MIA proper: **score exactly the tables the
generator measured, on the generator's own bins**, because those are the only
places a record's +1 count survives into the release.  What changes is how the
attacker learns the tables and the bins, and v2 keeps the two threat models on
separate, labelled paths (Steven, 2026-09-23):

cliques
    "public"    the star (1-way + gene x label).  Fixed by published config for
                every structure, so this is black box by construction.
    "recovered" black box: the star plus the tree, re-selected from the
                synthetic release by running the generator's own selection
                (`dp_select_tree`) on it with the noise switched off.  Inside a
                PGM fitted to star + tree, genes are conditionally independent
                given the label except along the measured edges, so the
                release's best tree is the measured tree up to sampling noise.
    "aux"       black box, route (a) in its simplest form: the generator's own
                tree selection run with no noise on the adversary's auxiliary
                pool (the candidate pool, the same data p_aux comes from), i.e.
                one noiseless shadow.  The selection follows population
                correlations, so this finds the tree even when the release
                carries it too weakly for "recovered" (tree_label, 2026-09-23:
                70-83% recall vs 0-1%).  MAMA-MIA proper's many-shadow
                frequency weighting is the extension.
    "true"      white box, a DIAGNOSTIC: the target's actual clique list, read
                from generator.pt.  Never a headline number.

edges
    "recovered" black box: re-run the generator's binning step on the release
                with the noise switched off -- per-gene quantiles for
                dp_quantile / quantile, the release's range split evenly for
                dp_uniform.  Rows are dithered uniformly inside their bin, so
                at high epsilon these land on the target's edges.
    "known"     white box, a DIAGNOSTIC: the target's fitted edges.
    "aux"       v1's choice: equal-frequency edges over the candidate pool.
    For `binning=uniform` the grid is public, so every choice uses it and the
    attack stays black box.

Whenever the attacker re-runs part of the generator's pipeline it does so with
no noise (epsilon -> infinity): it is trying to be as accurate as possible,
not private.

Score: for every chosen table and candidate x, log p_syn(x's cell) /
p_aux(x's cell), with a pseudo-count `alpha` on both, averaged over tables;
`class_centre` then subtracts each subtype's mean score (the subtype is known
and predicts the score but not membership).  `family_scores` exposes each
family (1way, gl, tree) separately, so aggregation arms are free to compute.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import datasets as D
from .. import targets as T
from .base import Attack, register
from .mahalamia import sigmoid_calibrate
from .mamamia import digitize, quantile_bin_edges

LABEL = "__label__"
FAMILIES = ("1way", "gl", "tree")


def center_by_class(s: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = s.astype(float).copy()
    for c in np.unique(y):
        k = y == c
        out[k] -= out[k].mean()
    return out


def _load_generator(dataset: str, generator: str, split: int):
    from mia import paths
    from mia.generators.pgm import PGMGenerator
    return PGMGenerator().load(paths.target_dir(dataset, generator, split)
                               / "generator.pt")._gen


def _gene_pos(dataset: str) -> dict:
    return {f"gene_{i}": i for i in range(len(D.gene_names(dataset)))}


def recover_edges(Xs: np.ndarray, params: dict, kind: str, X_pool: np.ndarray,
                  dataset: str, generator: str, split: int) -> np.ndarray:
    """Interior bin edges, shape (n_genes, n_bins-1), for the requested access."""
    K = int(params["n_bins"])
    binning = params.get("binning", "quantile")
    if binning == "uniform":                          # public grid: no choice
        lo, hi = params.get("bin_range", (0.0, 24.0))
        return np.tile(np.linspace(lo, hi, K + 1)[1:-1], (Xs.shape[1], 1))
    if kind == "aux":
        return quantile_bin_edges(X_pool, K)
    if kind == "known":
        gen = _load_generator(dataset, generator, split)
        pos = _gene_pos(dataset)
        out = np.empty((Xs.shape[1], K - 1))
        for j, g in enumerate(gen.selected_gene_names):
            e = np.asarray(gen._discretizer._edges[j])[1:-1]
            # Non-DP quantile binning merges tied edges, leaving some genes with
            # fewer bins; repeating the last edge keeps the same partition.
            out[pos[g]] = np.concatenate([e, np.repeat(e[-1], K - 1 - len(e))])
        return out
    if kind != "recovered":
        raise ValueError(f"unknown edges {kind!r}")
    if binning in ("quantile", "dp_quantile"):
        return quantile_bin_edges(Xs, K)
    if binning == "dp_uniform":
        lo, hi = Xs.min(axis=0), Xs.max(axis=0)
        return np.stack([np.linspace(a, b, K + 1)[1:-1] for a, b in zip(lo, hi)])
    raise ValueError(f"no black-box edge recovery for binning={binning!r}")


def recover_tree(Bs: np.ndarray, ys: np.ndarray, K: int, n_classes: int,
                 with_label: bool) -> list:
    """The generator's tree selection, run on the release with no noise."""
    from mia.generators.pgm import _import_upstream
    _import_upstream()
    from marginal_selection import dp_select_tree
    G = Bs.shape[1]
    codes = (np.arange(G) * K * n_classes)[None, :] + Bs * n_classes + ys[:, None]
    gl = np.bincount(codes.ravel(), minlength=G * K * n_classes)
    gl = gl.reshape(G, K, n_classes).astype(np.float64)
    pairs, _ = dp_select_tree(Bs, ys, gl, rho=1e12, rng=np.random.default_rng(0),
                              with_label=with_label)
    return [tuple(sorted(p)) for p in pairs]


def true_tree(dataset: str, generator: str, split: int) -> list:
    """White box: the target's measured gene-gene pairs, as column indices."""
    gen = _load_generator(dataset, generator, split)
    pos = _gene_pos(dataset)
    m = gen._marginals
    out = []
    for c in list(m.get("2way", [])) + list(m.get("3way", [])):
        genes = [g for g in c if g in pos]
        if len(genes) == 2:
            out.append(tuple(sorted(pos[g] for g in genes)))
    return out


def _log_ratio(codes_real, codes_syn, codes_aux, n_tables, n_cells, alpha):
    """log p_syn/p_aux at each real row's cell, per table.  codes: (n, T)."""
    off = (np.arange(n_tables, dtype=np.int64) * n_cells)[None, :]
    size = n_tables * n_cells

    def probs(c):
        cnt = np.bincount((c + off).ravel(), minlength=size).astype(np.float64)
        cnt = cnt.reshape(n_tables, n_cells) + alpha
        return cnt / cnt.sum(axis=1, keepdims=True)

    ps, pa = probs(codes_syn), probs(codes_aux)
    t = np.arange(n_tables)[None, :]
    return np.log(ps[t, codes_real]) - np.log(pa[t, codes_real])


@dataclass
class MAMAMIAv2(Attack):
    cliques: str = "recovered"
    edges: str = "recovered"
    families: tuple = FAMILIES
    class_centre: bool = True
    alpha: float = 0.5
    calibrate: bool = True

    name = "mamamia_v2"

    def tag(self) -> str:
        fam = "" if tuple(self.families) == FAMILIES else "_" + "+".join(self.families)
        cc = "_cc" if self.class_centre else ""
        return f"cl-{self.cliques}_ed-{self.edges}{fam}{cc}"

    @staticmethod
    def access(cliques: str, edges: str, binning: str) -> str:
        """'white-box' if anything came from inside the target, else 'black-box'."""
        wb = cliques == "true" or (edges == "known" and binning != "uniform")
        return "white-box" if wb else "black-box"

    def family_scores(self, dataset: str, generator: str, split: int) -> dict:
        """Per-family mean log-ratio for every candidate, plus what was used."""
        ncl = D.n_classes(dataset)
        Xr = D.load_expression(dataset).values.astype(np.float64)
        yr = D.encode_subtypes(dataset, D.load_subtypes(dataset).values)
        tg = T.load_target(dataset, generator, split)
        Xs, ys = tg["X"].astype(np.float64), np.asarray(tg["y_int"]).astype(np.int64)
        params = T.target_record(dataset, generator, split)["params"]
        K = int(params["n_bins"])
        structure = params.get("structure", "hierarchical")

        E = recover_edges(Xs, params, self.edges, Xr, dataset, generator, split)
        Br, Bs = digitize(Xr, E, K), digitize(Xs, E, K)
        G = Br.shape[1]

        if self.cliques == "true":
            tree = true_tree(dataset, generator, split)
        elif self.cliques == "recovered" and structure in ("tree", "tree_label"):
            tree = recover_tree(Bs, ys, K, ncl, structure == "tree_label")
        elif self.cliques == "aux" and structure in ("tree", "tree_label"):
            tree = recover_tree(Br, yr, K, ncl, structure == "tree_label")
        elif self.cliques in ("public", "recovered", "aux"):
            tree = []
        else:
            raise ValueError(f"unknown cliques {self.cliques!r}")
        tree_label = structure == "tree_label"

        out = {}
        if "1way" in self.families:
            out["1way"] = _log_ratio(Br, Bs, Br, G, K, self.alpha).mean(1)
        if "gl" in self.families:
            out["gl"] = _log_ratio(Br * ncl + yr[:, None], Bs * ncl + ys[:, None],
                                   Br * ncl + yr[:, None], G, K * ncl,
                                   self.alpha).mean(1)
        if "tree" in self.families and tree:
            a = np.array([p[0] for p in tree])
            b = np.array([p[1] for p in tree])
            ncell = K * K * (ncl if tree_label else 1)

            def code(B, y):
                c = B[:, a] * K + B[:, b]
                return c * ncl + y[:, None] if tree_label else c
            out["tree"] = _log_ratio(code(Br, yr), code(Bs, ys), code(Br, yr),
                                     len(tree), ncell, self.alpha).mean(1)
        return {"scores": out, "tree": tree, "edges": E, "y": yr,
                "n_tables": {"1way": G, "gl": G, "tree": len(tree)},
                "access": self.access(self.cliques, self.edges,
                                      params.get("binning", "quantile"))}

    def score(self, dataset: str, generator: str, split: int) -> np.ndarray:
        fs = self.family_scores(dataset, generator, split)
        n = fs["n_tables"]
        used = list(fs["scores"])
        w = np.array([n[f] for f in used], dtype=float)
        raw = sum(wi * fs["scores"][f] for wi, f in zip(w, used)) / w.sum()
        if self.class_centre:
            raw = center_by_class(raw, fs["y"])
        raw = np.nan_to_num(raw, nan=float(np.nanmedian(raw)))
        return sigmoid_calibrate(raw, log_transform=False) if self.calibrate else raw

    def params(self) -> dict:
        d = super().params()
        d["families"] = list(self.families)
        return d


register(MAMAMIAv2)
