"""The five roles a model plays in a membership-inference experiment.

The attack pipeline is usually described as a stack of models, but they are not
five *kinds* of model -- they are five *jobs*, and the same artifact can do a
different job in a different experiment.  Naming the jobs explicitly is what
lets the zoo be reused without anyone losing track of what is allowed.

    TARGET          the model we want to learn about.  Trained on real data we
                    are not supposed to know.  We only see what it released.

    BASE_SHADOW     trained on a real split we *do* control, purely so that it
                    can emit a synthetic dataset with known membership labels.
                    Under a white-box threat this role disappears entirely:
                    there is no released-synthetic-data layer to imitate.

    SYNTH_SHADOW    trained on a BASE_SHADOW's synthetic output, not on real
                    data.  This is the point of synth-shadow modelling: the
                    model whose losses train the meta-classifier must have been
                    fitted to synthetic data, because the model whose losses it
                    will be *applied* to -- the proxy -- was too.  Its
                    membership labels are inherited from its base shadow's real
                    split, which the zoo recomputes from the stored closure.

    INTERNAL_PROXY  a proxy trained on a *held-out* base shadow's synthetic
                    data, standing in for the target.  Because we know its
                    labels, it measures how the final proxy will behave under
                    matched conditions.  In the competition this was the only
                    way to estimate performance at all.  Here we can score the
                    final proxy directly, so this role has a narrower but still
                    essential job: it is the only honest signal for choosing
                    hyperparameters and ensemble weights.  Selecting those
                    against the final proxy's labels would be tuning on the
                    evaluation set.

    FINAL_PROXY     trained on the target's released synthetic data.  Our best
                    available stand-in for the target itself, and the model the
                    reported scores actually come from.  Under a white-box
                    threat this role also disappears -- we would use the target.

White-box, for reference, keeps only TARGET and a real-data shadow population:
no BASE_SHADOW, no synth-shadow layer, no proxy.  Black-box uses all five.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from . import ids as I
from . import registry as Z

TARGET = "target"
BASE_SHADOW = "base_shadow"
SYNTH_SHADOW = "synth_shadow"
INTERNAL_PROXY = "internal_proxy"
FINAL_PROXY = "final_proxy"

ROLES = (TARGET, BASE_SHADOW, SYNTH_SHADOW, INTERNAL_PROXY, FINAL_PROXY)

#: Roles that see data the adversary is not supposed to have.  Exactly one
#: artifact may hold the target role, and nothing the adversary builds may
#: descend from it.
ADVERSARY_ROLES = (BASE_SHADOW, SYNTH_SHADOW, INTERNAL_PROXY, FINAL_PROXY)


@dataclass
class Assignment:
    """Which artifacts play which role in one experiment."""
    experiment: str
    dataset: str
    roles: dict = field(default_factory=lambda: {r: [] for r in ROLES})

    def add(self, role: str, *artifact_ids: str) -> "Assignment":
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}; expected one of {ROLES}")
        self.roles.setdefault(role, []).extend(artifact_ids)
        return self

    def ids(self, role: str) -> list:
        return list(self.roles.get(role, []))

    def to_dict(self) -> dict:
        return {"experiment": self.experiment, "dataset": self.dataset,
                "roles": {r: self.ids(r) for r in ROLES if self.ids(r)}}


def check(a: Assignment) -> list[str]:
    """Every way this assignment could be leaking, as a list of complaints.

    An empty list means the experiment is clean.  These are cheap string and
    set comparisons on the index, so they are worth running on every experiment
    rather than only when something looks wrong.
    """
    problems: list[str] = []

    # 1. One artifact, one job.  Reuse across experiments is the whole point;
    #    reuse across roles inside one experiment is contamination.
    seen: dict[str, str] = {}
    for role in ROLES:
        for aid in a.ids(role):
            if aid in seen:
                problems.append(
                    f"artifact {aid[:8]} holds two roles in one experiment: "
                    f"{seen[aid]} and {role}")
            seen[aid] = role

    targets = a.ids(TARGET)
    if len(targets) > 1:
        problems.append(f"{len(targets)} artifacts claim the target role; expected 1")
    target = Z.get(targets[0]) if targets else None

    if target is not None:
        # 2. Nothing the adversary builds may descend from the target.  This is
        #    the failure that would look like a spectacular attack result.
        for role in ADVERSARY_ROLES:
            for aid in a.ids(role):
                if role == FINAL_PROXY:
                    continue            # the final proxy is *supposed* to
                if target.id in Z.ancestors(aid):
                    problems.append(
                        f"{role} {aid[:8]} descends from the target {target.id[:8]}")

        # 3. A base shadow trained on exactly the target's real samples would
        #    hand the meta-classifier the target's own membership labels.
        for aid in a.ids(BASE_SHADOW):
            art = Z.get(aid)
            if art is not None and art.closure == target.closure:
                problems.append(
                    f"base_shadow {aid[:8]} has the target's training set "
                    f"(closure {art.closure[:8]}) -- its labels are the target's")

        # 4. The final proxy must actually be trained on the target's output.
        for aid in a.ids(FINAL_PROXY):
            if target.id not in Z.ancestors(aid):
                problems.append(
                    f"final_proxy {aid[:8]} is not descended from the target "
                    f"{target.id[:8]} -- it is modelling something else")

    # 5. Internal proxies exist to be held out.  If one shares a base shadow
    #    with a synth-shadow in the same experiment, the meta-classifier has
    #    already seen the data it is being validated against.
    synth_bases = {b for aid in a.ids(SYNTH_SHADOW) for b in Z.ancestors(aid)}
    for aid in a.ids(INTERNAL_PROXY):
        shared = set(Z.ancestors(aid)) & synth_bases
        if shared:
            problems.append(
                f"internal_proxy {aid[:8]} shares ancestry {sorted(s[:8] for s in shared)} "
                f"with a synth_shadow -- it is not held out")

    # 6. Dataset agreement: a cohort mismatch silently scores the wrong samples.
    for role in ROLES:
        for aid in a.ids(role):
            art = Z.get(aid)
            if art is not None and art.dataset != a.dataset:
                problems.append(
                    f"{role} {aid[:8]} is from {art.dataset}, not {a.dataset}")

    return problems


def assert_clean(a: Assignment) -> None:
    problems = check(a)
    if problems:
        raise AssertionError(
            f"contaminated experiment {a.experiment!r}:\n  " + "\n  ".join(problems))


def reusable_as(role: str, dataset: str, generator: str | None = None,
                exclude_closures: Iterable[str] = ()) -> list:
    """Artifacts already in the zoo that could fill `role` in a new experiment.

    This is the payoff of separating roles from artifacts: a synth-shadow built
    for one trial is, mechanically, a fitted generator over synthetic data, and
    nothing stops another trial from treating its output as a target -- provided
    its training closure is not one this experiment must keep secret.
    """
    excluded = set(exclude_closures)
    want_kind = "sample" if role == TARGET else "fit"
    out = []
    for art in Z.find(dataset=dataset, kind=want_kind):
        if generator and art.generator != generator:
            continue
        if art.closure in excluded:
            continue
        out.append(art)
    return out
