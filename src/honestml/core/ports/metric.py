"""The ``Metric`` port.

One Scorer abstraction drives selection, HPO, ensembling and the leaderboard
uniformly. ``needs`` declares what projection of the model output the metric
consumes, so ``predict_proba_positive`` (hard-binary) is unnecessary: the metric
asks for ``proba``/``class``/``value`` and the use-case projects accordingly.
``value`` covers regression. ``sample_weight`` is a first-class, optional
argument — fixing it now avoids a breaking signature change later.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

import numpy as np

MetricNeeds = Literal["proba", "threshold", "class", "value"]


@runtime_checkable
class Metric(Protocol):
    """A scorer with a declared optimization direction and required input form.

    ``average`` is the multiclass averaging mode (``"macro"``/``"micro"``/
    ``"weighted"``); ``None`` means binary / metric default. Additive: a metric
    carries it as an instance field and ``score`` does not change signature.

    ``proper_proba`` flags a **proper probabilistic loss** (log-loss/Brier)
    whose value is changed by post-hoc calibration — the only metrics refinement-based
    selection may rank on. Ranking/argmax metrics (roc_auc/accuracy) carry ``False`` and
    are a no-op under refinement *by this gate*, not by any monotonicity assumption.
    """

    name: str
    greater_is_better: bool
    needs: MetricNeeds
    optimum: float
    average: str | None
    proper_proba: bool

    def score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> float: ...
