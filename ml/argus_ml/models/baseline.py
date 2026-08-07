"""Logistic regression baseline — Course 1, and the model we must beat.

Two reasons this exists beyond nostalgia:

1. **Nothing ships without beating a baseline.** A PR-AUC of 0.62 is
   meaningless in isolation. It is meaningful when logistic regression gets
   0.41 on the same split.
2. **Risk teams genuinely deploy it.** Adverse-action notices and model-risk
   review both favour a model whose coefficients are directly readable, so
   logistic regression is frequently the production challenger, not a toy.

Regularisation, feature scaling and class weighting are all applied here
exactly as taught: scaling because gradient descent on unscaled velocity
counts and log-amounts converges badly, L2 because the one-hot blocks are
collinear by construction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from argus_ml.models.base import RiskModel


class LogisticBaseline(RiskModel):
    """Scaled, L2-regularised logistic regression with class balancing."""

    name = "logistic_baseline"

    def __init__(self, C: float = 0.1, max_iter: int = 2_000, seed: int = 42) -> None:
        super().__init__()
        self.C = C
        self.max_iter = max_iter
        self.seed = seed
        self.pipeline: Pipeline | None = None
        self.feature_names: list[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> LogisticBaseline:
        self.feature_names = list(feature_names)
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=self.C,
                        max_iter=self.max_iter,
                        # Reweights the loss so the 0.5% positive class is not
                        # simply ignored by the optimiser.
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=self.seed,
                    ),
                ),
            ]
        )
        self.pipeline.fit(X, y)
        self._fitted = True
        self.metadata = self._build_metadata(
            version="v1",
            feature_names=self.feature_names,
            y=y,
            hyperparameters={"C": self.C, "class_weight": "balanced"},
            notes="Interpretable baseline. All other models must beat this on PR-AUC.",
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._require_fitted()
        assert self.pipeline is not None
        return self.pipeline.predict_proba(X)[:, 1]

    def explain(self, X: np.ndarray, top_k: int = 6) -> list[list[tuple[str, float, float]]]:
        """Attribution = scaled feature value x coefficient.

        For a linear model this *is* the exact contribution to the logit, so
        no approximation is involved — a genuine advantage over the tree
        ensemble when explaining a decision to a customer.
        """
        self._require_fitted()
        assert self.pipeline is not None
        scaler: StandardScaler = self.pipeline.named_steps["scaler"]
        clf: LogisticRegression = self.pipeline.named_steps["clf"]
        Xs = scaler.transform(X)
        coefs = clf.coef_[0]

        out: list[list[tuple[str, float, float]]] = []
        for i in range(len(X)):
            contribs = Xs[i] * coefs
            idx = np.argsort(-np.abs(contribs))[:top_k]
            out.append(
                [
                    (self.feature_names[j], float(X[i, j]), float(contribs[j]))
                    for j in idx
                ]
            )
        return out

    def coefficients(self) -> dict[str, float]:
        """Global coefficient table — goes straight into the model card."""
        self._require_fitted()
        assert self.pipeline is not None
        clf: LogisticRegression = self.pipeline.named_steps["clf"]
        return dict(
            sorted(
                zip(self.feature_names, (float(c) for c in clf.coef_[0]), strict=False),
                key=lambda kv: -abs(kv[1]),
            )
        )

    def get_params(self) -> dict[str, Any]:
        return {"C": self.C, "max_iter": self.max_iter, "seed": self.seed}


__all__ = ["LogisticBaseline"]
