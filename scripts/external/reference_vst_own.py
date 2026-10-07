"""VST fitted on the COMBINED reference set alone (824 samples), with the
challenge's recipe.  What an adversary holding only raw counts for the
auxiliary samples would compute; the challenge instead distributed the
reference set transformed jointly with the candidates.

Run with ~/.venvs/pydeseq2/bin/python after tcga_vst.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vst as VST  # noqa: E402

OUT = Path("~/data/TCGA_GDC/processed").expanduser()


def main() -> None:
    from pydeseq2.dds import DeseqDataSet
    d = np.load(OUT / "COMBINED_reference_counts.npz", allow_pickle=True)
    counts = pd.DataFrame(d["counts"], index=d["gene_ids"], columns=d["aliquots"])
    counts = counts[~counts.index.duplicated()]
    keep = VST.gene_filter(counts.values)
    c = counts.loc[keep]
    meta = pd.DataFrame({"project": [p.replace("-", "_") for p in d["project"]]}, index=c.columns)
    dds = DeseqDataSet(counts=c.T, metadata=meta, design="~project", quiet=True)
    dds.vst_fit(use_design=True)
    V = pd.DataFrame(dds.vst_transform(), index=c.columns, columns=c.index)
    genes = list(pd.read_csv(OUT / "COMBINED_reference_tpm.tsv", sep="\t", index_col=0, nrows=1).columns)
    V[genes].to_csv(OUT / "COMBINED_reference_vst_own.tsv", sep="\t")
    print(f"{int(keep.sum())} genes kept, trend {list(map(float, dds.uns['vst_trend_coeffs']))}")


if __name__ == "__main__":
    main()
