"""Content addressing for the model zoo.

Every artifact the pipeline produces is a deterministic function of the things
that went into it: a generator family, its hyperparameters, the data it was fit
to, and a seed.  Hashing that tuple gives an identifier that is stable across
machines and runs, which is what makes an artifact *reusable* -- two experiments
that ask for the same thing get the same object rather than each building their
own copy.

The hash has to be canonical or it is worthless: `{"a": 1, "b": 2}` and
`{"b": 2, "a": 1}` describe the same generator and must hash the same.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

HASH_LEN = 16


def _default(o: Any):
    """Make numpy scalars and Paths JSON-encodable without changing their value."""
    if hasattr(o, "item"):          # numpy scalar
        return o.item()
    if hasattr(o, "tolist"):        # numpy array
        return o.tolist()
    return str(o)


def canonical(obj: Any) -> str:
    """A JSON encoding where equal objects always produce equal strings."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_default)


def digest(obj: Any, length: int = HASH_LEN) -> str:
    return hashlib.sha1(canonical(obj).encode()).hexdigest()[:length]


def closure_hash(sample_ids: Iterable) -> str:
    """Identify a *set* of real samples, independent of order.

    This is the field contamination checks are built on: two artifacts trained
    on the same real samples have the same closure hash however they were
    otherwise produced, so a shadow that accidentally shares the target's
    training set is caught by comparing one string.
    """
    joined = "\x00".join(sorted(str(s) for s in sample_ids))
    return hashlib.sha1(joined.encode()).hexdigest()[:HASH_LEN]


# ── Data references ──────────────────────────────────────────────────────────
# A data reference names what an artifact was fit to.  It is either a slice of
# a real cohort or another artifact's output, which is what lets provenance be
# walked all the way back to real samples.

def real_ref(dataset: str, split: int, side: str = "train") -> str:
    return f"real:{dataset}:split{split}:{side}"


def subset_ref(dataset: str, closure: str) -> str:
    """A shadow's resample of the real cohort, named by which samples it holds."""
    return f"subset:{dataset}:{closure}"


def artifact_ref(artifact_id: str) -> str:
    return f"artifact:{artifact_id}"


def parse_ref(ref: str) -> tuple:
    """(kind, rest) for a data reference; kind is real / subset / artifact."""
    kind, _, rest = ref.partition(":")
    return kind, rest
