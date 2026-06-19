"""Numerical and physical metrics for low-rank industrial surrogates."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rel_fro_error(A: np.ndarray, Ahat: np.ndarray) -> float:
    """Relative Frobenius reconstruction error."""
    denom = np.linalg.norm(A, ord="fro")
    if denom <= 0 or not np.isfinite(denom):
        return float("nan")
    return float(np.linalg.norm(A - Ahat, ord="fro") / denom)


def rel_l2_error(x: np.ndarray, xhat: np.ndarray) -> float:
    """Relative vector 2-norm error."""
    denom = np.linalg.norm(x)
    if denom <= 0 or not np.isfinite(denom):
        return float("nan")
    return float(np.linalg.norm(x - xhat) / denom)


def safe_cond(D: np.ndarray) -> float:
    """Condition number with protection against singular or failed SVDs."""
    try:
        s = np.linalg.svd(D, compute_uv=False)
        if len(s) == 0 or s[-1] <= 1e-14:
            return float("inf")
        return float(s[0] / s[-1])
    except np.linalg.LinAlgError:
        return float("inf")


def confidence_half_width(values: pd.Series) -> float:
    """Normal-approximation 95% confidence half-width."""
    vals = values.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(vals) <= 1:
        return 0.0
    return float(1.96 * vals.std(ddof=1) / np.sqrt(len(vals)))


def storage_ratio_factorized(m: int, n: int, r: int, method: str) -> float:
    """Stored coefficients divided by dense snapshot storage."""
    if "SVD" in method or "POD" in method:
        stored = m * r + r + r * n
    elif "Mean" in method:
        stored = m
    else:
        stored = m * r + r * r + r * n
    return float(stored / max(1, m * n))


def energy_norm_error(K: np.ndarray, u_true: np.ndarray, u_hat: np.ndarray) -> float:
    """Relative error in the K-energy norm for SPD operators."""
    e = u_true - u_hat
    num = float(e.T @ K @ e)
    den = float(u_true.T @ K @ u_true)
    if den <= 0:
        return float("nan")
    return float(np.sqrt(max(num, 0.0) / den))


def equilibrium_residual(K: np.ndarray, u_hat: np.ndarray, f: np.ndarray) -> float:
    """Relative operator residual ||K u_hat - f|| / ||f||."""
    denom = np.linalg.norm(f)
    if denom <= 0:
        return float("nan")
    return float(np.linalg.norm(K @ u_hat - f) / denom)


def physical_response_metrics(K: np.ndarray, u_true: np.ndarray, u_hat: np.ndarray, f: np.ndarray) -> dict[str, float]:
    """Physical errors used across structural, thermal, and grid cases."""
    max_true = float(np.max(np.abs(u_true)))
    max_hat = float(np.max(np.abs(u_hat)))
    energy_true = float(0.5 * u_true.T @ K @ u_true)
    energy_hat = float(0.5 * u_hat.T @ K @ u_hat)
    return {
        "rel_state_error": rel_l2_error(u_true, u_hat),
        "rel_max_state_error": abs(max_true - max_hat) / max(max_true, 1e-30),
        "rel_energy_error": abs(energy_true - energy_hat) / max(abs(energy_true), 1e-30),
        "rel_energy_norm_error": energy_norm_error(K, u_true, u_hat),
        "rel_operator_residual": equilibrium_residual(K, u_hat, f),
        "true_max_state": max_true,
        "estimated_max_state": max_hat,
        "true_energy": energy_true,
        "estimated_energy": energy_hat,
    }
