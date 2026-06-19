"""Sparse sensor selection and reconstruction."""

from __future__ import annotations

import numpy as np


def select_variance_sensors(A_train: np.ndarray, sensor_count: int) -> np.ndarray:
    """Choose rows with largest training variance."""
    row_var = np.var(A_train, axis=1)
    order = np.argsort(row_var)[::-1]
    return np.sort(order[:sensor_count])


def select_random_sensors(n_rows: int, sensor_count: int, seed: int) -> np.ndarray:
    """Random sensor ablation."""
    rng = np.random.default_rng(50_000 + seed)
    return np.sort(rng.choice(n_rows, size=min(sensor_count, n_rows), replace=False))


def reconstruct_from_sensors(mean: np.ndarray, basis: np.ndarray, sensor_rows: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    """Recover a full state from sparse measurements and a learned basis."""
    if basis.shape[1] == 0:
        return mean[:, 0].copy()
    B_s = basis[sensor_rows, :]
    y_c = y - mean[sensor_rows, 0]
    lhs = B_s.T @ B_s + ridge * np.eye(basis.shape[1])
    rhs = B_s.T @ y_c
    coeff = np.linalg.solve(lhs, rhs)
    return mean[:, 0] + basis @ coeff
