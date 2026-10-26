"""Classical multi-fidelity baselines for comparison with residual SDNs."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import randomized_svd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


@dataclass
class MethodTiming:
    """Wall-clock training and inference times in seconds."""

    fit_s: float
    predict_s: float
    predict_per_sample_s: float


def _as_numpy(array) -> np.ndarray:
    if hasattr(array, "detach"):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _relative_l2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true))


def _relative_max(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.max(np.abs(y_true - y_pred)) / np.max(np.abs(y_true)))


def time_method(
    fit_fn,
    predict_fn,
    n_test: int,
    n_warmup: int = 3,
    n_repeat: int = 20,
) -> MethodTiming:
    """Benchmark fit once and average repeated prediction calls."""
    start = time.perf_counter()
    model = fit_fn()
    fit_s = time.perf_counter() - start

    for _ in range(n_warmup):
        predict_fn(model)

    start = time.perf_counter()
    for _ in range(n_repeat):
        predict_fn(model)
    predict_s = (time.perf_counter() - start) / n_repeat

    return MethodTiming(
        fit_s=fit_s,
        predict_s=predict_s,
        predict_per_sample_s=predict_s / max(n_test, 1),
    )


class LinearMFCorrector:
    """Linear map from LF reduced coefficients to the HF-LF residual."""

    def __init__(self, alpha: float = 1e-6):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha, fit_intercept=True)

    def fit(self, v_lf, u_lf, u_hf) -> LinearMFCorrector:
        v_lf = _as_numpy(v_lf)
        u_lf = _as_numpy(u_lf)
        u_hf = _as_numpy(u_hf)
        delta = u_hf - u_lf
        self.model.fit(v_lf, delta)
        return self

    def predict(self, v_lf, u_lf) -> np.ndarray:
        v_lf = _as_numpy(v_lf)
        u_lf = _as_numpy(u_lf)
        return u_lf + self.model.predict(v_lf)


class PODGPRResidualCorrector:
    """Nonlinear GP correction on POD coefficients of the HF-LF residual."""

    def __init__(self, rank: int = 10, alpha: float = 1e-4):
        self.rank = rank
        self.alpha = alpha
        self.modes: np.ndarray | None = None
        self.models: list[GaussianProcessRegressor] = []

    def fit(self, v_lf, u_lf, u_hf) -> PODGPRResidualCorrector:
        v_lf = _as_numpy(v_lf)
        u_lf = _as_numpy(u_lf)
        u_hf = _as_numpy(u_hf)
        delta = u_hf - u_lf

        self.modes, _, _ = randomized_svd(delta.T, n_components=self.rank, n_iter="auto")
        delta_pod = delta @ self.modes

        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(nu=2.5) + WhiteKernel(
            noise_level=self.alpha, noise_level_bounds=(1e-8, 1e-1)
        )
        self.models = []
        for kk in range(self.rank):
            gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True)
            gp.fit(v_lf, delta_pod[:, kk])
            self.models.append(gp)
        return self

    def predict(self, v_lf, u_lf) -> np.ndarray:
        if self.modes is None:
            raise RuntimeError("Call fit before predict.")

        v_lf = _as_numpy(v_lf)
        u_lf = _as_numpy(u_lf)
        delta_pod = np.column_stack([gp.predict(v_lf) for gp in self.models])
        delta = delta_pod @ self.modes.T
        return u_lf + delta


class CoKrigingMFCorrector:
    """
    Autoregressive co-kriging on HF POD coefficients:

    z_HF(μ) ≈ ρ z_LF(μ) + GP(μ),
    """

    def __init__(self, rank: int = 10, alpha: float = 1e-4):
        self.rank = rank
        self.alpha = alpha
        self.modes: np.ndarray | None = None
        self.rho: np.ndarray | None = None
        self.models: list[GaussianProcessRegressor] = []
        self.mu_scaler = StandardScaler()

    def fit(self, mu, u_lf, u_hf) -> CoKrigingMFCorrector:
        mu = _as_numpy(mu)
        if mu.ndim == 1:
            mu = mu.reshape(-1, 1)
        u_lf = _as_numpy(u_lf)
        u_hf = _as_numpy(u_hf)

        self.modes, _, _ = randomized_svd(u_hf.T, n_components=self.rank, n_iter="auto")
        z_lf = u_lf @ self.modes
        z_hf = u_hf @ self.modes

        mu_scaled = self.mu_scaler.fit_transform(mu)
        self.rho = np.zeros(self.rank)
        self.models = []

        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(nu=2.5) + WhiteKernel(
            noise_level=self.alpha, noise_level_bounds=(1e-8, 1e-1)
        )
        for kk in range(self.rank):
            lf_var = np.var(z_lf[:, kk])
            if lf_var > 1e-12:
                self.rho[kk] = np.cov(z_hf[:, kk], z_lf[:, kk])[0, 1] / lf_var
            else:
                self.rho[kk] = 0.0

            discrepancy = z_hf[:, kk] - self.rho[kk] * z_lf[:, kk]
            gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True)
            gp.fit(mu_scaled, discrepancy)
            self.models.append(gp)
        return self

    def predict(self, mu, u_lf) -> np.ndarray:
        if self.modes is None or self.rho is None:
            raise RuntimeError("Call fit before predict.")

        mu = _as_numpy(mu)
        if mu.ndim == 1:
            mu = mu.reshape(-1, 1)
        u_lf = _as_numpy(u_lf)
        mu_scaled = self.mu_scaler.transform(mu)

        z_lf = u_lf @ self.modes
        z_hf = np.zeros_like(z_lf)
        for kk, gp in enumerate(self.models):
            z_hf[:, kk] = self.rho[kk] * z_lf[:, kk] + gp.predict(mu_scaled)

        return z_hf @ self.modes.T


def _relative_errors_per_sample(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column relative L2 and max errors."""
    y_true = _as_numpy(y_true)
    y_pred = _as_numpy(y_pred)
    l2 = np.linalg.norm(y_true - y_pred, axis=0) / np.linalg.norm(y_true, axis=0)
    max_err = np.max(np.abs(y_true - y_pred), axis=0) / np.max(np.abs(y_true), axis=0)
    return l2, max_err


def evaluate_mf_methods(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    reference_lf: np.ndarray | None = None,
) -> dict[str, dict[str, float | np.ndarray]]:
    """Compute aggregate and per-sample relative L2 and max norms."""
    y_true = _as_numpy(y_true)
    metrics: dict[str, dict[str, float | np.ndarray]] = {}

    if reference_lf is not None:
        l2_ps, max_ps = _relative_errors_per_sample(y_true, _as_numpy(reference_lf))
        metrics["LF"] = {
            "l2": float(np.mean(l2_ps)),
            "max": float(np.mean(max_ps)),
            "l2_per_sample": l2_ps,
            "max_per_sample": max_ps,
        }

    for name, pred in predictions.items():
        pred = _as_numpy(pred)
        l2_ps, max_ps = _relative_errors_per_sample(y_true, pred)
        metrics[name] = {
            "l2": float(np.mean(l2_ps)),
            "max": float(np.mean(max_ps)),
            "l2_per_sample": l2_ps,
            "max_per_sample": max_ps,
        }
    return metrics
