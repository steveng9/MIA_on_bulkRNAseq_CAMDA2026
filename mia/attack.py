"""Unified MIA entry point.

Usage
-----
# Full pipeline (default: synth-shadow mode, CVAE, K=15, Q=5)
python -m mia.attack --model cvae --dataset BRCA

# Generate K+Q synthetic datasets only (no attack)
python -m mia.attack --model cvae --generate-synthetic --k 20 --q 5

# Also run real-shadow evaluation + domain gap
python -m mia.attack --model cvae --real

# ND attack (limited to K+Q ≤ 5 in synth mode)
python -m mia.attack --model nd --k 3 --q 2

# Force re-run specific stages
python -m mia.attack --model cvae --force features,classifier

# Skip challenge submission
python -m mia.attack --model cvae --no-submission
"""

import sys
import argparse

sys.stdout.reconfigure(line_buffering=True)

from . import config
from .backends import NDBackend, CVAEBackend
from .pipeline import (
    run_pipeline,
    ensure_splits,
    ensure_real_shadows,
    ensure_synthetic,
    _hdr,
)


def _build_backend(model: str, dataset: str):
    if model == "nd":
        return NDBackend(dataset)
    if model == "cvae":
        return CVAEBackend(dataset)
    raise ValueError(f"Unknown model '{model}'.  Choose 'nd' or 'cvae'.")


def _apply_profile(model: str, profile: str):
    if profile is None:
        return
    if model == "cvae":
        config.apply_cvae_profile(profile)
    else:
        config.apply_profile(profile)


def _check_nd_synth_limit(k: int, q: int):
    """ND synth mode is capped at 5 total splits (Blue Team repo)."""
    from .backends.nd import _ND_MAX_SYNTH_SPLITS
    n_total = k + q
    if n_total > _ND_MAX_SYNTH_SPLITS:
        print(f"  WARNING: ND synth mode supports at most {_ND_MAX_SYNTH_SPLITS} total splits "
              f"(K+Q={n_total} requested).  Capping to K+Q={_ND_MAX_SYNTH_SPLITS}.")
        excess = n_total - _ND_MAX_SYNTH_SPLITS
        q      = max(1, q - excess)
        k      = _ND_MAX_SYNTH_SPLITS - q
        print(f"  Using K={k}  Q={q}")
    return k, q


def main():
    parser = argparse.ArgumentParser(
        description="Unified MIA pipeline (NoisyDiffusion and CVAE)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",   choices=["nd", "cvae"], default="cvae",
                        help="Generative model to attack")
    parser.add_argument("--dataset", choices=["BRCA", "COMBINED"], default="BRCA")
    parser.add_argument("--k",  type=int, default=None,
                        help="Shadow models for MLP training (default: config.DEFAULT_K)")
    parser.add_argument("--q",  type=int, default=None,
                        help="Holdout models for internal evaluation (default: config.DEFAULT_Q)")
    parser.add_argument("--real", action="store_true",
                        help="Also run real-shadow eval + domain-gap measurement")
    parser.add_argument("--no-submission", action="store_true",
                        help="Skip challenge prediction step")
    parser.add_argument("--generate-synthetic", action="store_true",
                        help="Generate K+Q synthetic datasets only, then exit (no attack)")
    parser.add_argument("--profile", default=None,
                        help="Named config profile (baseline | tuned)")
    parser.add_argument("--label-mode", choices=["real", "knn", "none"], default=None,
                        help="CVAE label source (overrides config; default: real)")
    parser.add_argument("--classifier", choices=["mlp", "rf"], default="mlp",
                        help="Classifier type: 'mlp' (default) or 'rf' (Random Forest)")
    parser.add_argument("--force", default="",
                        help="Comma-separated stages to force re-run: "
                             "splits, real_shadows, synthetic, synth_shadows, "
                             "features, classifier, submission, shadows (=real_shadows+synth_shadows), all")
    parser.add_argument("--device", default=config.DEVICE)
    args = parser.parse_args()

    # ── Apply config overrides ────────────────────────────────────────────────
    _apply_profile(args.model, args.profile)
    if args.label_mode:
        config.CVAE_LABEL_MODE = args.label_mode
    config.CLASSIFIER_TYPE = args.classifier

    force_stages = set()
    if args.force:
        for s in args.force.split(","):
            force_stages.add(s.strip())

    k = args.k if args.k is not None else config.DEFAULT_K
    q = args.q if args.q is not None else config.DEFAULT_Q

    if args.model == "nd" and not args.real:
        k, q = _check_nd_synth_limit(k, q)

    backend = _build_backend(args.model, args.dataset)

    print(f"[mia.attack]  model={args.model}  dataset={args.dataset}  "
          f"K={k}  Q={q}  classifier={config.CLASSIFIER_TYPE}  "
          f"label_mode={config.CVAE_LABEL_MODE}  device={args.device}")
    if args.model == "cvae":
        print(f"             profile={config.CVAE_ACTIVE_PROFILE}  "
              f"feature_mode={config.CVAE_FEATURE_MODE}  "
              f"temp_list={config.CVAE_TEMP_LIST}  n_samples={config.CVAE_N_SAMPLES}")
    else:
        print(f"             profile={config.ACTIVE_PROFILE}  "
              f"feature_mode={config.FEATURE_MODE}  "
              f"T_LIST={config.T_LIST}  N_NOISE={config.N_NOISE}")

    # ── Standalone synthetic generation (--generate-synthetic) ───────────────
    if args.generate_synthetic:
        n_total    = k + q
        all_splits = list(range(1, n_total + 1))
        _hdr(f"GENERATE SYNTHETIC ONLY — {args.model.upper()} / {args.dataset}  "
             f"K={k}  Q={q}  total={n_total}")
        ensure_splits(args.dataset, n_total, force="splits" in force_stages)
        ensure_real_shadows(backend, all_splits, args.device,
                            force="real_shadows" in force_stages or "shadows" in force_stages)
        ensure_synthetic(backend, all_splits, args.device,
                         force="synthetic" in force_stages)
        print(f"\nDone.  {n_total} synthetic datasets ready in "
              f"mia_output/pipeline/synthetic/{args.model}/{args.dataset}/")
        return

    # ── Full pipeline ─────────────────────────────────────────────────────────
    run_pipeline(
        backend      = backend,
        k            = k,
        q            = q,
        real_mode    = args.real,
        device       = args.device,
        no_submission= args.no_submission,
        force_stages = force_stages,
    )


if __name__ == "__main__":
    main()
