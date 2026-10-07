"""Download GDC STAR-count files for the TCGA projects of the challenge cohorts.

Writes, under --out (default ~/data/TCGA_GDC):

  gdc_star_counts_manifest.csv   one row per open STAR-counts file
  parts/<batch>.npz              unstranded counts and TPM, genes x files
  genes.tsv                      gene_id, gene_name, gene_type (row order)

The raw per-file TSVs (4 MB each) are parsed in memory and not kept.  Batches
that already exist are skipped, so the script resumes.

    python scripts/external/gdc_download.py --workers 6
"""

from __future__ import annotations

import argparse
import io
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROJECTS = ["TCGA-BRCA", "TCGA-KIRC", "TCGA-LUAD", "TCGA-LUSC", "TCGA-PRAD",
            "TCGA-COAD", "TCGA-OV", "TCGA-LIHC", "TCGA-KIRP", "TCGA-ESCA",
            "TCGA-PAAD", "TCGA-SKCM"]
API = "https://api.gdc.cancer.gov"


def fetch_manifest() -> pd.DataFrame:
    filters = {"op": "and", "content": [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": PROJECTS}},
        {"op": "=", "content": {"field": "data_type", "value": "Gene Expression Quantification"}},
        {"op": "=", "content": {"field": "analysis.workflow_type", "value": "STAR - Counts"}},
        {"op": "=", "content": {"field": "access", "value": "open"}}]}
    fields = ("file_id,file_name,file_size,cases.project.project_id,cases.submitter_id,"
              "cases.samples.sample_type,cases.samples.submitter_id,"
              "cases.samples.portions.analytes.aliquots.submitter_id")
    r = requests.post(f"{API}/files", json={"filters": filters, "size": 20000,
                                            "format": "JSON", "fields": fields}, timeout=300)
    r.raise_for_status()
    rows = []
    for h in r.json()["data"]["hits"]:
        c = h["cases"][0]
        s = c["samples"][0]
        rows.append(dict(
            file_id=h["file_id"], file_name=h["file_name"], size=h["file_size"],
            project=c["project"]["project_id"], patient=c["submitter_id"],
            sample=s["submitter_id"], sample_type=s["sample_type"],
            aliquot=s["portions"][0]["analytes"][0]["aliquots"][0]["submitter_id"]))
    return pd.DataFrame(rows).sort_values("file_id").reset_index(drop=True)


def fetch_batch(batch_id: int, file_ids: list, out: Path) -> str:
    dest = out / "parts" / f"{batch_id:04d}.npz"
    if dest.exists():
        return f"{batch_id} cached"
    for attempt in range(6):
        try:
            r = requests.post(f"{API}/data", json={"ids": file_ids}, timeout=1800)
            r.raise_for_status()
            if len(file_ids) == 1:          # a single id comes back bare, not as a tar
                files = [(f"{file_ids[0]}/x.tsv", io.BytesIO(r.content))]
            else:
                tf = tarfile.open(fileobj=io.BytesIO(r.content))
                files = [(m.name, tf.extractfile(m)) for m in tf.getmembers()
                         if m.name.endswith(".tsv")]
            counts, tpm, got, genes = [], [], [], None
            for name, handle in files:
                member = type("M", (), {"name": name})
                df = pd.read_csv(handle, sep="\t", comment="#")
                df = df[df.gene_id.str.startswith("ENSG")]
                if genes is None:
                    genes = df[["gene_id", "gene_name", "gene_type"]]
                elif not np.array_equal(genes.gene_id.values, df.gene_id.values):
                    raise RuntimeError(f"gene order differs in {member.name}")
                counts.append(df["unstranded"].to_numpy(np.int32))
                tpm.append(df["tpm_unstranded"].to_numpy(np.float32))
                got.append(member.name.split("/")[0])
            if sorted(got) != sorted(file_ids):
                raise RuntimeError(f"batch {batch_id}: {len(got)} of {len(file_ids)} files")
            genes_path = out / "genes.tsv"
            if not genes_path.exists():
                genes.to_csv(genes_path, sep="\t", index=False)
            tmp = dest.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, file_ids=np.array(got), counts=np.stack(counts, 1),
                                tpm=np.stack(tpm, 1), gene_ids=genes.gene_id.values)
            tmp.rename(dest)
            return f"{batch_id} ok ({len(got)} files)"
        except Exception as exc:  # network errors are routine on a 24 GB pull
            time.sleep(20 * (attempt + 1))
            last = exc
    return f"{batch_id} FAILED: {last}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="~/data/TCGA_GDC")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--batch", type=int, default=40)
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    (out / "parts").mkdir(parents=True, exist_ok=True)

    manifest_path = out / "gdc_star_counts_manifest.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
    else:
        manifest = fetch_manifest()
        manifest.to_csv(manifest_path, index=False)
    ids = sorted(manifest.file_id)
    batches = [ids[i:i + args.batch] for i in range(0, len(ids), args.batch)]
    print(f"{len(ids)} files in {len(batches)} batches", flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        for msg in pool.map(lambda kv: fetch_batch(kv[0], kv[1], out), enumerate(batches)):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
