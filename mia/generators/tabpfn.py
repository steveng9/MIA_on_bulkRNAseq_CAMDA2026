"""TabPFN target generator: synthetic data from a tabular foundation model
(Hollmann et al., Nature 2025), through Prior Labs' own unsupervised extension
(`tabpfn_extensions.unsupervised.TabPFNUnsupervisedModel`).

Nothing is trained.  The pretrained transformer is used in context: columns are
generated one at a time, each sampled from TabPFN's predictive distribution
given the columns already generated, with the real training rows as the
context set.  That makes it an unusual membership target -- the "model" the
attacker faces is the training table itself, held in a prompt.

Defaults are the extension's: temperature 1.0 and 3 conditioning permutations
per column.  The class label is column 0 and is flagged categorical, so labels
are generated first and every gene is conditioned on the subtype.

Cost is one in-context fit per (column, permutation), each over all earlier
columns, so 978 genes means ~2,900 forward passes with up to 978 features.
`preprocess` is where that is controlled: "none" (the default; TabPFN applies
its own scaling) generates gene by gene, while e.g. "standard+pca:64" has it
generate 64 component scores instead, ~15x cheaper and inside the feature
counts TabPFN was pretrained on, at the price of a rank-64 release.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import preprocessing as pp
from .base import Generator, register


@dataclass
class TabPFNGenerator(Generator):
    preprocess: str = "none"
    temperature: float = 1.0
    n_permutations: int = 3
    n_estimators: int | None = None      # None: TabPFN's default ensemble size
    model_version: str = "default"       # or e.g. "v2", "v2.5": tabpfn's ModelVersion

    name = "tabpfn"
    requires = {"tabpfn": "6.0", "tabpfn-extensions": "0.5"}
    env = "sota"

    def __post_init__(self):
        self._model = None
        self.scaler = None
        self.n_classes = None

    def _estimators(self):
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        kw = {"device": self.device, "random_state": self.seed,
              "ignore_pretraining_limits": True}
        if self.n_estimators is not None:
            kw["n_estimators"] = int(self.n_estimators)
        if self.model_version != "default":
            from tabpfn.constants import ModelVersion
            v = ModelVersion(self.model_version)
            return (TabPFNClassifier.create_default_for_version(v, **kw),
                    TabPFNRegressor.create_default_for_version(v, **kw))
        return TabPFNClassifier(**kw), TabPFNRegressor(**kw)

    def fit(self, X: np.ndarray, y: np.ndarray, n_classes: int) -> "TabPFNGenerator":
        import copy
        import os
        os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
        # The current weights are gated behind a licence key.  It is read from
        # TABPFN_TOKEN, or from a private file kept outside the repository.
        token_file = Path(os.environ.get("CAMDA_TABPFN_TOKEN_FILE",
                                         "~/.config/camda/tabpfn_token")).expanduser()
        if "TABPFN_TOKEN" not in os.environ and token_file.exists():
            os.environ["TABPFN_TOKEN"] = token_file.read_text().strip()
        import torch
        from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel

        class _OnCPU(TabPFNUnsupervisedModel):
            # The extension keeps the table being filled on the CPU but, with a
            # GPU model, gets regressor logits back on the GPU and then fails
            # writing the sampled column.  Bring each prediction to the CPU.
            def sample_from_model_prediction_(self, *a, **k):
                pred, sampled = super().sample_from_model_prediction_(*a, **k)
                if isinstance(pred, dict):
                    pred = dict(pred)
                    pred["logits"] = torch.as_tensor(pred["logits"]).detach().cpu()
                    pred["criterion"] = copy.deepcopy(pred["criterion"]).cpu()
                else:
                    pred = pred.cpu()
                return pred, sampled.cpu()

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        self.n_classes = int(n_classes)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.scaler, Xs = pp.fit_scaler(self.preprocess, X, y)
        table = np.column_stack([y.astype(np.float32), Xs])
        clf, reg = self._estimators()
        self._model = _OnCPU(tabpfn_clf=clf, tabpfn_reg=reg)
        self._model.set_categorical_features([0])
        self._model.fit(torch.tensor(table, dtype=torch.float32))
        return self

    def sample(self, n: int) -> tuple:
        out = self._model.generate_synthetic_data(
            n_samples=int(n), t=self.temperature, n_permutations=self.n_permutations)
        out = out.cpu().numpy() if hasattr(out, "cpu") else np.asarray(out)
        y_syn = np.clip(np.rint(out[:, 0]), 0, self.n_classes - 1).astype(np.int64)
        return pp.invert_scaler(self.scaler, out[:, 1:], y_syn), y_syn


register(TabPFNGenerator)
