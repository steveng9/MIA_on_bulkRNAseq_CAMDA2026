"""Build MeLoMIA proxy features for some (generator, split) cells and stop.

Scoring trains one proxy per target, in sequence; for TabSyn that is about an
hour each.  This builds them ahead, so several processes on different GPUs can
share the work (each cell is one cached file), and the scoring run that
follows only reads them.

    python scripts/melomia_proxies.py <config> <attack-label> --generators G... --splits 1 2
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mia.experiment import Experiment   # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("label")
    p.add_argument("--generators", nargs="+", required=True)
    p.add_argument("--splits", nargs="+", type=int, required=True)
    args = p.parse_args()
    exp = Experiment.load(args.config)
    attack = exp.build_attack(args.label)
    for g in args.generators:
        for s in args.splits:
            attack._ensure_proxy_features(exp.dataset, g, s)
            print(f"proxy features ready: {exp.dataset}/{g}/split_{s}", flush=True)


if __name__ == "__main__":
    main()
