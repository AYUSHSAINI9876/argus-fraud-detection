"""Gaussian anomaly detection — Course 3, implemented from the maths.

Why an unsupervised layer sits beside a supervised model at all:

The XGBoost model can only catch fraud that resembles fraud it has *seen
labelled*. Labels arrive via chargebacks, weeks late. A novel attack pattern
launched today is, by definition, absent from the training set — the
supervised model will score it confidently benign.

Density estimation has no such blind spot. It learns what *normal* looks
like from the overwhelming majority class and flags whatever sits in a
low-probability region, regardless of whether that pattern has ever been
labelled. The two models fail in uncorrelated ways, which is exactly why
running both beats running either.

Implemented directly rather than via `sklearn.EllipticEnvelope` because the
multivariate Gaussian fit and the epsilon-selection procedure are the
substance of the Course 3 material, and because the per-feature contribution
breakdown below falls out of the maths for free.
"""

from __future__ import annotations

import numpy as np

from argus_ml.models.base import RiskModel


class GaussianAnomalyDetector(RiskModel):
    """Multivariate Gaussian density estimator with F1-selected epsilon.

    Fits mu and sigma on *legitimate traffic only*, then scores every
    transaction by its density under that distribution. Low density = anomaly.
    """

    name = "gaussian_anomaly"

    def __init__(self, use_multivariate: bool = True, eps_reg: float = 1e-6) -> None:
        super().__init__()
        self.use_multivariate = use_multivariate
        self.eps_reg = eps_reg
        self.mu: np.ndarray | None = None
        self.sigma2: np.ndarray | None = None       # univariate variances
        self.cov: np.ndarray | None = None          # covariance matrix
        self.cov_inv: np.ndarray | None = None
        self.log_det: float = 0.0
        self.epsilon: float = 0.0
        self.feature_names: list[str] = []
        self._log_p_ref: np.ndarray | None = None   # for score normalisation

    # -- fitting ----------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> GaussianAnomalyDetector:
        """Estimate the normal distribution from non-fraud rows only.

        `y` is used *only* to exclude known fraud from the fit and to tune
        epsilon on the validation split — the model itself is unsupervised.
        """
        self.feature_names = list(feature_names)
        X_normal = X[y == 0]

        self.mu = X_normal.mean(axis=0)
        centred = X_normal - self.mu

        if self.use_multivariate:
            cov = np.cov(centred, rowvar=False)
            # Ridge on the diagonal: the one-hot blocks are rank-deficient by
            # construction, so the raw covariance is singular.
            cov += np.eye(cov.shape[0]) * self.eps_reg * np.trace(cov) / cov.shape[0]
            self.cov = cov
            sign, self.log_det = np.linalg.slogdet(cov)
            if sign <= 0:
                # Fall back rather than emit NaNs on a degenerate matrix.
                self.use_multivariate = False
            else:
                self.cov_inv = np.linalg.inv(cov)

        if not self.use_multivariate:
            self.sigma2 = centred.var(axis=0) + self.eps_reg

        # Choose epsilon on a labelled validation slice by maximising F1 —
        # the selection procedure taught in Course 3.
        if X_val is not None and y_val is not None:
            self.epsilon = self._select_epsilon(X_val, y_val)

        # Reference distribution of log-densities, used to squash the raw
        # density into a comparable 0-1 score for the UI.
        self._log_p_ref = self._log_density(X_normal[: min(len(X_normal), 50_000)])

        self._fitted = True
        self.metadata = self._build_metadata(
            version="v1",
            feature_names=self.feature_names,
            y=y,
            hyperparameters={
                "use_multivariate": self.use_multivariate,
                "epsilon": float(self.epsilon),
                "fit_rows_normal_only": int(len(X_normal)),
            },
            notes="Unsupervised layer. Catches novel typologies absent from labelled data.",
        )
        return self

    def _log_density(self, X: np.ndarray) -> np.ndarray:
        """Log p(x) under the fitted Gaussian.

        Computed in log space throughout — in 60 dimensions the raw density
        underflows float64 immediately.
        """
        assert self.mu is not None
        d = X.shape[1]
        centred = X - self.mu

        if self.use_multivariate and self.cov_inv is not None:
            # -0.5 * (d log 2pi + log|S| + (x-mu)^T S^-1 (x-mu))
            #
            # Written as a matmul plus a row-wise sum rather than
            # np.einsum("ij,jk,ik->i", ...). The einsum form is the obvious
            # transcription of the maths but is dramatically slower here: it
            # falls back to a naive nested loop instead of dispatching to
            # BLAS. On a million-row scoring pass that is minutes versus
            # seconds, for identical output.
            maha = ((centred @ self.cov_inv) * centred).sum(axis=1)
            return -0.5 * (d * np.log(2 * np.pi) + self.log_det + maha)

        assert self.sigma2 is not None
        return -0.5 * np.sum(
            np.log(2 * np.pi * self.sigma2) + (centred**2) / self.sigma2, axis=1
        )

    def _select_epsilon(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Sweep candidate thresholds, keep the one with the best F1."""
        log_p = self._log_density(X_val)
        lo, hi = np.percentile(log_p, 0.01), np.percentile(log_p, 20.0)
        best_f1, best_eps = 0.0, float(lo)

        for eps in np.linspace(lo, hi, 200):
            pred = log_p < eps
            tp = float(np.sum(pred & (y_val == 1)))
            fp = float(np.sum(pred & (y_val == 0)))
            fn = float(np.sum(~pred & (y_val == 1)))
            if tp == 0:
                continue
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)
            if f1 > best_f1:
                best_f1, best_eps = f1, float(eps)
        return best_eps

    # -- scoring ----------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Anomaly score in [0, 1], where 1 is maximally anomalous.

        This is *not* a probability of fraud and is deliberately not treated
        as one — it feeds the analyst UI as a separate signal and enters the
        decision policy as its own term.
        """
        self._require_fitted()
        log_p = self._log_density(X)
        assert self._log_p_ref is not None
        # Percentile rank against normal traffic, inverted.
        ranks = np.searchsorted(np.sort(self._log_p_ref), log_p) / len(self._log_p_ref)
        return np.clip(1.0 - ranks, 0.0, 1.0)

    def is_anomaly(self, X: np.ndarray) -> np.ndarray:
        self._require_fitted()
        return self._log_density(X) < self.epsilon

    def explain(self, X: np.ndarray, top_k: int = 6) -> list[list[tuple[str, float, float]]]:
        """Which dimensions drove the anomaly.

        Per-feature standardised squared deviation. For the diagonal model
        this decomposes the log-density exactly; for the full-covariance model
        it is an approximation that ignores cross terms, which is acceptable
        for ranking the top drivers in a UI.
        """
        self._require_fitted()
        assert self.mu is not None
        if self.sigma2 is not None:
            var = self.sigma2
        else:
            assert self.cov is not None
            var = np.diag(self.cov)

        centred = X - self.mu
        z2 = (centred**2) / var

        out: list[list[tuple[str, float, float]]] = []
        for i in range(len(X)):
            idx = np.argsort(-z2[i])[:top_k]
            out.append(
                [(self.feature_names[j], float(X[i, j]), float(z2[i, j])) for j in idx]
            )
        return out


__all__ = ["GaussianAnomalyDetector"]
