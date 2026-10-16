"""Refit only the denoiser of MeLoMIA-TabSyn synth-shadow probes.

The probes built before `probe_diff_warmup` existed mostly have a denoiser that
never trained (see docs/SOTA_GENERATORS.md, "Probe collapse").  Their VAEs are
fine and are most of the cost, so keep them: re-encode the internal synthetic
data, retrain the denoiser under the current probe recipe, and drop the cached
loss features so the stack re-extracts them.

    python scripts/repair_tabsyn_probes.py BRCA --worker 0 --of 4
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from mia import preprocessing as pp
from mia.attacks.melomia.backends import TabSynBackend
from mia.generators.tabsyn_diffusion import MLPDiffusion, Model

ap = argparse.ArgumentParser()
ap.add_argument("dataset")
ap.add_argument("--cache", default="n200")
ap.add_argument("--worker", type=int, default=0)
ap.add_argument("--of", type=int, default=1)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

root = Path("artifacts/attacks/melomia_tabsyn") / a.dataset / a.cache
be = TabSynBackend(dataset=a.dataset, device="cuda", seed=a.seed)
models = sorted(root.glob("synth_shadows/shadow_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
for path in models:
    k = int(path.stem.split("_")[1])
    done = path.with_suffix(".rewarmed")
    if k % a.of != a.worker or done.exists():
        continue
    g = be.probe()
    g.load(path)
    d = np.load(root / "internal_synth" / f"shadow_{k}.npz")
    Xs = pp.apply_scaler(g.scaler, d["X"].astype(np.float32)).astype(np.float32)
    z = g.encode(Xs, np.zeros(len(Xs), dtype=np.int64))
    torch.manual_seed(a.seed + 5000 + k)
    g.diffusion = Model(denoise_fn=MLPDiffusion(z.shape[1], g.dim_t), hid_dim=z.shape[1]).to(g.device)
    g._fit_diffusion(z)
    g.save(path)
    (root / "features" / f"shadow_{k}.npz").unlink(missing_ok=True)
    done.write_text(f"{g.diff_final_loss:.4f}\n")
    print(f"probe {k}: denoiser refitted, final loss {g.diff_final_loss:.3f}", flush=True)
