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
    "shadow"    black box, MAMA-MIA proper: `n_shadows` subsamples of a
                HELD-OUT auxiliary set, each run through the target's selection
                with no noise; every table is weighted by the fraction of
                shadows that chose it (`shadow_cliques`).  COMBINED uses the
                challenge's reference set (labels predicted by a classifier
                trained on the release); BRCA has none, so it uses the
                non-member candidates and is labelled optimistic.
    "true"      white box, a DIAGNOSTIC: the target's actual clique list, read
                from generator.pt.  Never a headline number.

For structure=forest (k_label, l_pairs) even the (gene, label) tables and the
1-way tables are data-dependent, so "recovered", "aux" and "shadow" re-run the
whole selection, and "public" falls back to guessing the full star.

edges
    "recovered" black box: re-run the generator's binning step on the release
                with the noise switched off -- per-gene quantiles for
                dp_quantile / quantile, the release's range split evenly for
                dp_uniform.  Rows are dithered uniformly inside their bin, so
                at high epsilon these land on the target's edges.
    "steps"     black box: the jumps in each gene's released density.  Rows
                are dithered uniformly inside their bin, so the density is
                flat within a bin and steps at every edge; the maximum-
                likelihood K-piece histogram finds them (`step_edges`).
                MEASURED WORSE than "recovered" (2026-09-24: edge MAE
                0.12-0.17 vs 0.02-0.04 at eps=1000).  dp_quantile interpolates
                linearly inside each public 0.5-wide grid cell, so adjacent
                bins in a cell have nearly equal density and their edges barely
                show as steps.  What that construction does give: inside a
                grid cell the edges form an exact arithmetic progression.
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
    if kind == "steps":
        return step_edges(Xs, K)
    if kind != "recovered":
        raise ValueError(f"unknown edges {kind!r}")
    if binning in ("quantile", "dp_quantile"):
        return quantile_bin_edges(Xs, K)
    if binning == "dp_uniform":
        lo, hi = Xs.min(axis=0), Xs.max(axis=0)
        return np.stack([np.linspace(a, b, K + 1)[1:-1] for a, b in zip(lo, hi)])
    raise ValueError(f"no black-box edge recovery for binning={binning!r}")


def _step_edges_1d(x: np.ndarray, K: int, n_cand: int,
                   min_frac: float = 1.0 / 3.0) -> np.ndarray:
    """K-1 interior edges of the best K-piece piecewise-uniform density for x.

    The generator dithers every synthetic value uniformly inside its bin, so
    each gene's released values have a density that is flat within a bin and
    jumps exactly at the generator's edges.  The maximum-likelihood K-piece
    histogram (free breakpoints) finds those jumps: segment s holding n_s of
    the points over width w_s contributes n_s log(n_s / w_s), and dynamic
    programming over candidate breakpoints maximises the sum.  Candidates are
    midpoints of the gaps between consecutive sorted values, thinned to
    `n_cand` by rank, then each chosen edge is refined over every gap between
    its neighbouring candidates.  Free-breakpoint histograms overfit by
    wrapping tiny segments around chance clusters, so every segment must hold
    at least `min_frac` of an equal-depth bin's share (n / K); dp_quantile's
    bins are equal-depth by design.
    """
    xs = np.sort(x)
    n = len(xs)
    gaps = (xs[:-1] + xs[1:]) / 2.0                     # n-1 possible edges
    idx = np.unique(np.linspace(0, n - 2, min(n_cand, n - 1)).round().astype(int))
    cand = np.concatenate([[xs[0]], gaps[idx], [xs[-1]]])
    cnt = np.concatenate([[0], idx + 1, [n]])           # points left of each candidate
    M = len(cand)
    i, j = np.triu_indices(M, k=1)
    ns = (cnt[j] - cnt[i]).astype(float)
    ws = np.maximum(cand[j] - cand[i], 1e-9)
    seg = np.full((M, M), -np.inf)
    m_min = max(1.0, min_frac * n / K)
    seg[i, j] = np.where(ns >= m_min, ns * np.log(np.maximum(ns, 1) / ws), -np.inf)
    best = seg[0].copy()                                # 1 segment ending at j
    back = []
    for _ in range(K - 1):
        tot = best[:, None] + seg
        arg = np.argmax(tot, axis=0)
        best = tot[arg, np.arange(M)]
        back.append(arg)
    # walk back from the last candidate
    cuts, j = [], M - 1
    for arg in reversed(back):
        j = arg[j]
        cuts.append(j)
    cuts = sorted(cuts)
    edges = cand[cuts].copy()
    # refine each edge over all gaps between its neighbouring candidates
    bounds = [0] + cuts + [M - 1]
    for e in range(len(cuts)):
        lo_c, hi_c = bounds[e], bounds[e + 2]
        a, b = cand[lo_c], cand[hi_c]
        na, nb = cnt[lo_c], cnt[hi_c]
        g_lo, g_hi = max(cnt[bounds[e]] , 1) - 1, min(cnt[hi_c], n - 1) - 1
        if g_hi <= g_lo:
            continue
        k = np.arange(g_lo, g_hi + 1)
        t = gaps[k]
        n1, n2 = (k + 1 - na).astype(float), (nb - (k + 1)).astype(float)
        m_min = max(1.0, min_frac * n / K)
        ll = (n1 * np.log(np.maximum(n1, 1) / np.maximum(t - a, 1e-9))
              + n2 * np.log(np.maximum(n2, 1) / np.maximum(b - t, 1e-9)))
        ll = np.where((n1 >= m_min) & (n2 >= m_min), ll, -np.inf)
        if not np.isfinite(ll).any():
            continue
        edges[e] = t[int(np.argmax(ll))]
    return np.sort(edges)


def step_edges(Xs: np.ndarray, K: int, n_cand: int = 300) -> np.ndarray:
    """`_step_edges_1d` for every gene.  Shape (n_genes, K-1)."""
    return np.stack([_step_edges_1d(Xs[:, g], K, n_cand) for g in range(Xs.shape[1])])


def _upstream_selection():
    from mia.generators.pgm import _import_upstream
    _import_upstream()
    import marginal_selection
    return marginal_selection


def _gl_counts(B: np.ndarray, y: np.ndarray, K: int, n_classes: int) -> np.ndarray:
    G = B.shape[1]
    codes = (np.arange(G) * K * n_classes)[None, :] + B * n_classes + y[:, None]
    gl = np.bincount(codes.ravel(), minlength=G * K * n_classes)
    return gl.reshape(G, K, n_classes).astype(np.float64)


def recover_tree(Bs: np.ndarray, ys: np.ndarray, K: int, n_classes: int,
                 with_label: bool) -> list:
    """The generator's tree selection, run on (Bs, ys) with no noise."""
    ms = _upstream_selection()
    pairs, _ = ms.dp_select_tree(Bs, ys, _gl_counts(Bs, ys, K, n_classes), rho=1e12,
                                 rng=np.random.default_rng(0), with_label=with_label)
    return [tuple(sorted(p)) for p in pairs]


def select_cliques(params: dict, B: np.ndarray, y: np.ndarray, K: int,
                   n_classes: int) -> dict:
    """Re-run the target's table selection on (B, y) with the noise switched off.

    Returns {"one": genes with a 1-way table, "gl": genes with a (gene, label)
    table, "pairs": (a, b) gene pairs, "pair_label": whether pairs carry the
    label}.  Used on the release ("recovered"), on the candidate pool ("aux")
    and on each shadow subsample ("shadow").
    """
    structure = params.get("structure", "hierarchical")
    G = B.shape[1]
    allg = list(range(G))
    if structure == "hierarchical":
        return {"one": allg, "gl": allg, "pairs": [], "pair_label": False}
    if structure in ("tree", "tree_label"):
        return {"one": allg, "gl": allg,
                "pairs": recover_tree(B, y, K, n_classes, structure == "tree_label"),
                "pair_label": structure == "tree_label"}
    if structure != "forest":
        raise ValueError(f"no selection model for structure={structure!r}")
    ms = _upstream_selection()
    k = min(int(params.get("k_label", G)), G)
    l = min(int(params.get("l_pairs", 0)), G - 1)
    lab = np.bincount(y, minlength=n_classes).astype(float)
    if 0 < k < G:
        genes_k, _ = ms.dp_select_label_genes(B, y, lab, k, K, rho=1e12,
                                              rng=np.random.default_rng(0))
    else:
        genes_k = list(range(k))
    pairs = []
    if l > 0:
        ref = np.tile(lab[None, None, :] / K, (G, K, 1))
        gl = _gl_counts(B, y, K, n_classes)
        ref[genes_k] = gl[genes_k]
        pairs, _ = ms.dp_select_tree(B, y, ref, rho=1e12, rng=np.random.default_rng(0),
                                     with_label=False, n_edges=l)
        pairs = [tuple(sorted(p)) for p in pairs]
    covered = set(genes_k) | {g for p in pairs for g in p}
    return {"one": [g for g in allg if g not in covered], "gl": sorted(genes_k),
            "pairs": pairs, "pair_label": False}


def true_cliques(dataset: str, generator: str, split: int) -> dict:
    """White box: the target's measured tables, as column indices."""
    gen = _load_generator(dataset, generator, split)
    pos = _gene_pos(dataset)
    m = gen._marginals
    one, gl, pairs, pair_label = [], [], [], False
    for c in m.get("1way", []):
        if c[0] in pos:
            one.append(pos[c[0]])
    for c in list(m.get("2way", [])) + list(m.get("3way", [])):
        genes = [pos[g] for g in c if g in pos]
        if len(genes) == 1:
            gl.append(genes[0])
        elif len(genes) == 2:
            pairs.append(tuple(sorted(genes)))
            pair_label = pair_label or len(c) == 3
    return {"one": one, "gl": gl, "pairs": pairs, "pair_label": pair_label}


def true_tree(dataset: str, generator: str, split: int) -> list:
    """White box: the target's measured gene-gene pairs, as column indices."""
    return true_cliques(dataset, generator, split)["pairs"]


def shadow_aux(dataset: str, split: int, Xs: np.ndarray, ys: np.ndarray):
    """The held-out auxiliary data for shadow selection, and what it is.

    COMBINED: the challenge's reference set of known non-members, which carries
    no subtype labels; the attacker labels it with a classifier trained on the
    synthetic release (which is labelled).  BRCA has no such set, so its
    non-member candidates stand in -- flagged "optimistic", since an attacker
    would not know which candidates are non-members (Steven, 2026-09-24).
    """
    ref = D.load_reference(dataset)
    if ref is not None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        Xa = ref[D.gene_names(dataset)].values.astype(np.float64) \
            if set(D.gene_names(dataset)) <= set(ref.columns) else ref.values.astype(np.float64)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
        clf.fit(Xs, ys)
        return Xa, clf.predict(Xa).astype(np.int64), "reference set (labels predicted from the release)"
    X = D.load_expression(dataset).values.astype(np.float64)
    y = D.encode_subtypes(dataset, D.load_subtypes(dataset).values)
    m = D.membership_labels(dataset, split).astype(bool)
    return X[~m], y[~m].astype(np.int64), "non-member candidates (OPTIMISTIC)"


def shadow_cliques(params: dict, Ba: np.ndarray, ya: np.ndarray, K: int,
                   n_classes: int, n_shadows: int, frac: float, seed: int = 0):
    """MAMA-MIA proper's shadow route: selection frequencies over subsamples.

    Each shadow is a `frac` subsample of the auxiliary rows, run through the
    target's selection with no noise.  Returns per-table weights in [0, 1]:
    the fraction of shadows that chose each table.
    """
    rng = np.random.default_rng(seed)
    n = len(Ba)
    w = {"one": {}, "gl": {}, "pairs": {}}
    pair_label = False
    for _ in range(n_shadows):
        idx = rng.choice(n, size=max(2, int(frac * n)), replace=False)
        c = select_cliques(params, Ba[idx], ya[idx], K, n_classes)
        pair_label = c["pair_label"]
        for fam in w:
            for t in c[fam]:
                w[fam][t] = w[fam].get(t, 0.0) + 1.0 / n_shadows
    return {"one": list(w["one"]), "gl": list(w["gl"]), "pairs": list(w["pairs"]),
            "pair_label": pair_label,
            "weights": {fam: np.array(list(w[fam].values())) for fam in w}}


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
    n_shadows: int = 20
    shadow_frac: float = 0.8

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
        """Per-family weighted mean log-ratio for every candidate, plus what was used."""
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
        allg = list(range(G))
        aux_note = ""

        if self.cliques == "true":
            cl = true_cliques(dataset, generator, split)
        elif self.cliques == "public":
            # The star is public for hierarchical/tree structures; for a
            # forest the chosen tables are hidden and the star is a guess.
            cl = {"one": allg, "gl": allg, "pairs": [], "pair_label": False}
        elif self.cliques == "recovered":
            cl = select_cliques(params, Bs, ys, K, ncl)
        elif self.cliques == "aux":
            cl = select_cliques(params, Br, yr, K, ncl)
        elif self.cliques == "shadow":
            Xa, ya, aux_note = shadow_aux(dataset, split, Xs, ys)
            cl = shadow_cliques(params, digitize(Xa, E, K), ya, K, ncl,
                                self.n_shadows, self.shadow_frac)
        else:
            raise ValueError(f"unknown cliques {self.cliques!r}")
        wts = cl.get("weights", {})

        def wmean(L, fam):
            w = wts.get(fam)
            if w is None:
                return L.mean(1), float(L.shape[1])
            return (L * w[None, :]).sum(1) / w.sum(), float(w.sum())

        out, n_tab = {}, {}
        if "1way" in self.families and cl["one"]:
            g = np.array(cl["one"])
            out["1way"], n_tab["1way"] = wmean(
                _log_ratio(Br[:, g], Bs[:, g], Br[:, g], len(g), K, self.alpha), "one")
        if "gl" in self.families and cl["gl"]:
            g = np.array(cl["gl"])
            out["gl"], n_tab["gl"] = wmean(
                _log_ratio(Br[:, g] * ncl + yr[:, None], Bs[:, g] * ncl + ys[:, None],
                           Br[:, g] * ncl + yr[:, None], len(g), K * ncl, self.alpha), "gl")
        tree = cl["pairs"]
        if "tree" in self.families and tree:
            a = np.array([p[0] for p in tree])
            b = np.array([p[1] for p in tree])
            pl = cl["pair_label"]
            ncell = K * K * (ncl if pl else 1)

            def code(B, y):
                c = B[:, a] * K + B[:, b]
                return c * ncl + y[:, None] if pl else c
            out["tree"], n_tab["tree"] = wmean(
                _log_ratio(code(Br, yr), code(Bs, ys), code(Br, yr), len(tree),
                           ncell, self.alpha), "pairs")
        access = self.access(self.cliques, self.edges, params.get("binning", "quantile"))
        if "OPTIMISTIC" in aux_note:
            access += " (optimistic aux)"
        return {"scores": out, "tree": tree, "cliques": cl, "edges": E, "y": yr,
                "n_tables": n_tab, "access": access, "aux": aux_note,
                "structure": structure}

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
