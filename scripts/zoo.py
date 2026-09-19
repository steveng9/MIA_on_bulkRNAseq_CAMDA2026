#!/usr/bin/env python
"""Inspect the model zoo.

    python scripts/zoo.py list --dataset BRCA
    python scripts/zoo.py list --dataset BRCA --generator cvae --kind fit
    python scripts/zoo.py show <artifact-id>
    python scripts/zoo.py lineage <artifact-id>
    python scripts/zoo.py reusable --dataset BRCA --role base_shadow
    python scripts/zoo.py verify                 # every recorded run's roles
    python scripts/zoo.py du                     # disk by generator and kind

An artifact id may be abbreviated to any unique prefix.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mia import paths                        # noqa: E402
from mia.zoo import registry as Z            # noqa: E402
from mia.zoo import roles as R               # noqa: E402


def _resolve(prefix: str) -> Z.Artifact:
    hits = [a for a in Z.all_artifacts() if a.id.startswith(prefix)]
    if not hits:
        raise SystemExit(f"no artifact starting with {prefix!r}")
    if len(hits) > 1:
        raise SystemExit(f"{prefix!r} is ambiguous: {[h.id[:12] for h in hits]}")
    return hits[0]


def _dir_bytes(d: Path) -> int:
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) if d.exists() else 0


def cmd_list(args) -> None:
    query = {k: v for k, v in
             (("dataset", args.dataset), ("generator", args.generator),
              ("kind", args.kind)) if v}
    arts = Z.find(**query)
    if not arts:
        print("no artifacts match")
        return
    print(f"{'id':<14} {'kind':<7} {'dataset':<9} {'generator':<6} "
          f"{'n':<7} {'closure':<10} {'created':<20} source")
    for a in arts:
        print(f"{a.id[:12]:<14} {a.kind:<7} {a.dataset:<9} {a.generator:<6} "
              f"{str(a.n or '-'):<7} {a.closure[:8]:<10} {a.created:<20} {a.source}")
    print(f"\n{len(arts)} artifacts")


def cmd_show(args) -> None:
    a = _resolve(args.id)
    print(json.dumps({**a.__dict__, "dir": str(a.dir),
                      "bytes": _dir_bytes(a.dir)}, indent=2))


def cmd_lineage(args) -> None:
    a = _resolve(args.id)
    print(Z.lineage(a.id))
    print(f"\ntraining closure: {a.n_closure} real samples ({a.closure[:8]})")


def cmd_reusable(args) -> None:
    arts = R.reusable_as(args.role, args.dataset, args.generator)
    for a in arts:
        print(f"{a.id[:12]}  {a.kind:<7} {a.generator:<6} "
              f"closure={a.closure[:8]} n_closure={a.n_closure}  {a.note}")
    print(f"\n{len(arts)} artifacts could fill role {args.role!r}")


def cmd_verify(args) -> None:
    """Re-check the role assignment of every run that recorded one."""
    bad = total = 0
    for d in sorted(paths.RUNS_DIR.glob("*")):
        rp = d / "roles.json"
        if not rp.exists():
            continue
        total += 1
        spec = json.loads(rp.read_text())
        a = R.Assignment(spec.get("experiment", d.name), spec["dataset"])
        for role, aids in spec.get("roles", {}).items():
            a.add(role, *aids)
        problems = R.check(a)
        if problems:
            bad += 1
            print(f"[CONTAMINATED] {d.name}")
            for p in problems:
                print(f"    {p}")
    print(f"\nchecked {total} runs with recorded roles; {bad} contaminated")
    if total == 0:
        print("(runs predating the zoo do not record roles -- nothing to check)")
    sys.exit(1 if bad else 0)


def cmd_du(args) -> None:
    rows = {}
    for a in Z.all_artifacts():
        key = (a.kind, a.generator)
        b = _dir_bytes(a.dir)
        n, tot = rows.get(key, (0, 0))
        rows[key] = (n + 1, tot + b)
    if not rows:
        print("zoo is empty")
        return
    print(f"{'kind':<8} {'generator':<10} {'count':>6} {'size':>10}")
    for (kind, gen), (n, tot) in sorted(rows.items(), key=lambda r: -r[1][1]):
        print(f"{kind:<8} {gen:<10} {n:>6} {tot / 1e9:>9.2f}G")
    print(f"{'total':<19} {sum(n for n, _ in rows.values()):>6} "
          f"{sum(t for _, t in rows.values()) / 1e9:>9.2f}G")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("list"); pl.set_defaults(fn=cmd_list)
    pl.add_argument("--dataset"); pl.add_argument("--generator")
    pl.add_argument("--kind", choices=Z.KINDS)

    ps = sub.add_parser("show"); ps.set_defaults(fn=cmd_show); ps.add_argument("id")
    pg = sub.add_parser("lineage"); pg.set_defaults(fn=cmd_lineage); pg.add_argument("id")

    pr = sub.add_parser("reusable"); pr.set_defaults(fn=cmd_reusable)
    pr.add_argument("--dataset", required=True)
    pr.add_argument("--role", required=True, choices=R.ROLES)
    pr.add_argument("--generator")

    sub.add_parser("verify").set_defaults(fn=cmd_verify)
    sub.add_parser("du").set_defaults(fn=cmd_du)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
