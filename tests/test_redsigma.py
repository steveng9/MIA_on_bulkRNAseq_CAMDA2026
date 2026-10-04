"""The `redsigma` port reproduces Tucker et al.'s submitted prediction files.

Their repo reports no metrics; what it ships is the eight prediction CSVs they
submitted to the challenge.  Regenerating those from the released challenge
data is the strongest available check that the attack here is the attack that
won.  Skipped when their repo or the challenge data is not on disk.
"""

import io
import zipfile

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from mia import paths
from mia.attacks.redsigma import redsigma_scores, rule_for

# challenge file number -> the rule their submission used for it
RULES = {1: "knn200", 2: "knn500", 3: "cvae", 4: "mvn"}
REF = (paths.CHALLENGE_DATA / "RED_TCGA-COMBINED"
       / "TCGA-COMBINED_primary_tumor_star_deseq_VST_lmgenes_reference.tsv")


def _zip(ds):
    return paths.REDSIGMA_REPO / "submissions" / f"redteam_RedSigma_TCGA-{ds}.zip"


def _red(ds):
    return paths.CHALLENGE_DATA / f"RED_TCGA-{ds}"


@pytest.mark.parametrize("ds", ["BRCA", "COMBINED"])
@pytest.mark.parametrize("i", [1, 2, 3, 4])
def test_reproduces_submitted_predictions(ds, i):
    syn = _red(ds) / f"synthetic_data_{i}.csv"
    if not (_zip(ds).exists() and syn.exists()):
        pytest.skip("RedSigma repo or challenge data not available")
    with zipfile.ZipFile(_zip(ds)) as z:
        want = pd.read_csv(io.BytesIO(
            z.read(f"synthetic_data_{i}_predictions.csv"))).iloc[:, 0].values

    X = pd.read_csv(_red(ds) / f"TCGA-{ds}_primary_tumor_star_deseq_VST_lmgenes.tsv",
                    sep="\t", index_col=0).T.values
    Xs = pd.read_csv(syn).values
    # as submitted: the official reference for COMBINED, none for BRCA
    Xr = (pd.read_csv(REF, sep="\t", index_col=0).T.values
          if ds == "COMBINED" else None)
    got = redsigma_scores(X, Xs, Xr, rule=RULES[i], combined=ds == "COMBINED")

    assert got.shape == want.shape
    assert spearmanr(got, want)[0] > 0.999999
    # BRCA's cvae rule divides rounding noise by 1e-10 (distance-to-self), so
    # its values agree to ~1e-3 relative rather than to machine precision.
    np.testing.assert_allclose(got, want, rtol=5e-3 if (ds, i) == ("BRCA", 3) else 1e-6)


def test_rule_for_follows_their_dispatcher():
    assert rule_for("nd") == "knn200"
    assert rule_for("pgg") == rule_for("pgm@epsilon=1") == "knn500"
    assert rule_for("cvae") == rule_for("dpcvae@epsilon=1000") == "cvae"
    assert rule_for("mvn") == "mvn"
    assert rule_for("tabsyn") == "mvn"      # unrecognised name: their else-branch
