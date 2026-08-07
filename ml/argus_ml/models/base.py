"""Common interface every Argus model implements.

Keeping one protocol means the serving layer, the evaluation harness and the
champion/challenger router never need to know which model family they hold.
Swapping logistic regression for XGBoost is a config change, not a rewrite.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np


@dataclass
class ModelMetadata:
    """Everything needed to reproduce and audit a trained artefact.

    `feature_names` is the load-bearing field: the serving path asserts the
    incoming vector matches this list exactly, so a feature added offline but
    not deployed online fails loudly at startup instead of silently scoring
    garbage.
    """

    name: str
    version: str
    trained_at: str
    feature_names: list[str]
    train_rows: int
    train_positive_rate: float
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    train_window: tuple[str, str] | None = None
    git_sha: str | None = None
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, default=str)


class RiskModel(ABC):
    """A scorer that maps a feature matrix to calibrated fraud probabilities."""

    name: str = "base"

    def __init__(self) -> None:
        self.metadata: ModelMetadata | None = None
        self._fitted: bool = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> RiskModel:
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(fraud) in [0, 1], one per row."""

    def explain(self, X: np.ndarray, top_k: int = 6) -> list[list[tuple[str, float, float]]]:
        """Per-row feature attributions as (name, value, contribution).

        Default implementation returns nothing; subclasses that support SHAP
        or coefficients override it. The API degrades gracefully — an analyst
        sees the score with a note that explanations are unavailable rather
        than the request failing.
        """
        return [[] for _ in range(len(X))]

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{self.name}: called before fit()")

    def save(self, path: str | Path) -> Path:
        """Persist model plus metadata side by side."""
        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        if self.metadata:
            path.with_suffix(".meta.json").write_text(
                self.metadata.to_json(), encoding="utf-8"
            )
        return path

    @staticmethod
    def load(path: str | Path) -> RiskModel:
        model = joblib.load(Path(path))
        if not isinstance(model, RiskModel):
            raise TypeError(f"{path} did not contain a RiskModel")
        return model

    def _build_metadata(
        self,
        version: str,
        feature_names: list[str],
        y: np.ndarray,
        hyperparameters: dict[str, Any],
        notes: str = "",
    ) -> ModelMetadata:
        return ModelMetadata(
            name=self.name,
            version=version,
            trained_at=datetime.now(UTC).isoformat(),
            feature_names=list(feature_names),
            train_rows=int(len(y)),
            train_positive_rate=float(np.mean(y)),
            hyperparameters=hyperparameters,
            notes=notes,
        )


__all__ = ["RiskModel", "ModelMetadata"]
