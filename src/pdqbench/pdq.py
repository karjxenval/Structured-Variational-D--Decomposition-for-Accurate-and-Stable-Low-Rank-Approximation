"""Baselines and core-conditioned PDQ factorization."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import solve

from .metrics import rel_fro_error, safe_cond, storage_ratio_factorized


@dataclass
class FittedLowRankModel:
    """A fitted surrogate basis with metadata for deployment evaluation."""

    method: str
    rank: int
    mean: np.ndarray
    basis: np.ndarray
    Ahat_train: np.ndarray
    time_sec: float
    kappa_D: float
    sweeps: int
    storage_ratio: float
    final_step: float
    final_obj_decrease: float
    details: Dict[str, np.ndarray]

    def reconstruct_full(self, A: np.ndarray, ridge: float = 1e-10) -> np.ndarray:
        """Best least-squares reconstruction of full snapshots using this basis."""
        if self.basis.shape[1] == 0:
            return np.repeat(self.mean, A.shape[1], axis=1)
        Ac = A - self.mean
        lhs = self.basis.T @ self.basis + ridge * np.eye(self.basis.shape[1])
        coeff = np.linalg.solve(lhs, self.basis.T @ Ac)
        return self.mean + self.basis @ coeff


def _center(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = A.mean(axis=1, keepdims=True)
    return A - mean, mean


def fit_mean_response(A: np.ndarray) -> FittedLowRankModel:
    """Industrial naive baseline: always predict the training mean response."""
    m, n = A.shape
    start = time.perf_counter()
    mean = A.mean(axis=1, keepdims=True)
    Ahat = np.repeat(mean, n, axis=1)
    elapsed = time.perf_counter() - start
    return FittedLowRankModel(
        "Mean response baseline",
        0,
        mean,
        np.zeros((m, 0)),
        Ahat,
        elapsed,
        np.nan,
        0,
        storage_ratio_factorized(m, n, 1, "Mean"),
        np.nan,
        np.nan,
        {},
    )


def truncated_svd_fit(A: np.ndarray, r: int) -> FittedLowRankModel:
    """POD / truncated SVD baseline on centered snapshots."""
    m, n = A.shape
    Ac, mean = _center(A)
    start = time.perf_counter()
    U, s, Vt = np.linalg.svd(Ac, full_matrices=False)
    rr = min(r, U.shape[1])
    basis = U[:, :rr]
    Ahat_c = (U[:, :rr] * s[:rr]) @ Vt[:rr, :]
    elapsed = time.perf_counter() - start
    return FittedLowRankModel(
        "Truncated SVD / POD",
        rr,
        mean,
        basis,
        mean + Ahat_c,
        elapsed,
        np.nan,
        0,
        storage_ratio_factorized(m, n, rr, "SVD"),
        np.nan,
        np.nan,
        {"singular_values": s[:rr], "Vt": Vt[:rr, :]},
    )


def randomized_svd_fit(A: np.ndarray, r: int, oversampling: int, n_iter: int, seed: int) -> FittedLowRankModel:
    """Randomized SVD baseline on centered snapshots."""
    rng = np.random.default_rng(20_000 + seed)
    m, n = A.shape
    Ac, mean = _center(A)
    ell = min(n, max(1, r + oversampling))
    start = time.perf_counter()
    Omega = rng.normal(size=(n, ell))
    Y = Ac @ Omega
    for _ in range(n_iter):
        Y = Ac @ (Ac.T @ Y)
    Qbasis, _ = np.linalg.qr(Y, mode="reduced")
    Bsmall = Qbasis.T @ Ac
    Ub, s, Vt = np.linalg.svd(Bsmall, full_matrices=False)
    U = Qbasis @ Ub
    rr = min(r, U.shape[1])
    basis = U[:, :rr]
    Ahat_c = (U[:, :rr] * s[:rr]) @ Vt[:rr, :]
    elapsed = time.perf_counter() - start
    return FittedLowRankModel(
        "Randomized SVD",
        rr,
        mean,
        basis,
        mean + Ahat_c,
        elapsed,
        np.nan,
        0,
        storage_ratio_factorized(m, n, rr, "SVD"),
        np.nan,
        np.nan,
        {"singular_values": s[:rr], "Vt": Vt[:rr, :]},
    )


def pdq_objective(Ac: np.ndarray, P: np.ndarray, D: np.ndarray, Q: np.ndarray,
                  alpha_P: float, alpha_D: float, alpha_Q: float) -> float:
    R = Ac - P @ D @ Q
    return float(
        0.5 * np.linalg.norm(R, ord="fro") ** 2
        + 0.5 * alpha_P * np.linalg.norm(P, ord="fro") ** 2
        + 0.5 * alpha_D * np.linalg.norm(D, ord="fro") ** 2
        + 0.5 * alpha_Q * np.linalg.norm(Q, ord="fro") ** 2
    )


def init_pdq_balanced(Ac: np.ndarray, r: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    U, s, Vt = np.linalg.svd(Ac, full_matrices=False)
    rr = min(r, len(s))
    U = U[:, :rr]
    s = np.maximum(s[:rr], 1e-14)
    Vt = Vt[:rr, :]
    root = np.sqrt(s)
    P = U * root
    D = np.eye(rr)
    Q = root[:, None] * Vt
    return P, D, Q


def init_pdq_random(Ac: np.ndarray, r: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(30_000 + seed)
    m, n = Ac.shape
    rr = min(r, m, n)
    scale = np.linalg.norm(Ac, ord="fro") / max(np.sqrt(m * n), 1.0)
    P = scale * rng.normal(size=(m, rr)) / np.sqrt(max(rr, 1))
    D = np.eye(rr)
    Q = rng.normal(size=(rr, n)) / np.sqrt(max(rr, 1))
    return P, D, Q


def solve_D_update(P: np.ndarray, Q: np.ndarray, Ac: np.ndarray, alpha_D: float) -> np.ndarray:
    r = P.shape[1]
    Gp = P.T @ P
    Gq = Q @ Q.T
    C = P.T @ Ac @ Q.T
    L = np.kron(Gq.T, Gp) + alpha_D * np.eye(r * r)
    d = solve(L, C.reshape(-1, order="F"), assume_a="pos")
    return d.reshape((r, r), order="F")


def clip_core_condition(D: np.ndarray, kappa_max: float) -> np.ndarray:
    U, s, Vt = np.linalg.svd(D, full_matrices=False)
    smax = max(float(s[0]), 1e-14)
    smin = smax / max(kappa_max, 1.0)
    return U @ np.diag(np.maximum(s, smin)) @ Vt


def fit_pdq_model(
    A: np.ndarray,
    r: int,
    alpha_P: float = 1e-4,
    alpha_D: float = 1e-4,
    alpha_Q: float = 1e-4,
    max_sweeps: int = 120,
    tol: float = 1e-7,
    conditioned_core: bool = False,
    kappa_core_max: float = 50.0,
    method_name: str = "PDQ ridge",
    init: str = "balanced",
    seed: int = 13,
) -> Tuple[FittedLowRankModel, pd.DataFrame]:
    """Fit A ≈ mean + P D Q by alternating ridge updates."""
    m, n = A.shape
    Ac, mean = _center(A)
    if init == "random":
        P, D, Q = init_pdq_random(Ac, r, seed)
    else:
        P, D, Q = init_pdq_balanced(Ac, r)
    rr = P.shape[1]
    obj_prev = pdq_objective(Ac, P, D, Q, alpha_P, alpha_D, alpha_Q)
    history = []
    start = time.perf_counter()

    for sweep in range(1, max_sweeps + 1):
        P_old, D_old, Q_old = P.copy(), D.copy(), Q.copy()

        HP = D @ Q @ Q.T @ D.T + alpha_P * np.eye(rr)
        P = Ac @ Q.T @ D.T @ np.linalg.inv(HP)

        D_candidate = solve_D_update(P, Q, Ac, alpha_D)
        if conditioned_core:
            D_projected = clip_core_condition(D_candidate, kappa_core_max)
            obj_before = pdq_objective(Ac, P, D_old, Q, alpha_P, alpha_D, alpha_Q)
            lam = 1.0
            for _ in range(25):
                D_try = (1.0 - lam) * D_old + lam * D_projected
                obj_try = pdq_objective(Ac, P, D_try, Q, alpha_P, alpha_D, alpha_Q)
                if obj_try <= obj_before or lam < 1e-8:
                    D = D_try
                    break
                lam *= 0.5
        else:
            D = D_candidate

        HQ = D.T @ P.T @ P @ D + alpha_Q * np.eye(rr)
        RHS = D.T @ P.T @ Ac
        Q = np.linalg.solve(HQ, RHS)

        Ahat_c = P @ D @ Q
        obj = pdq_objective(Ac, P, D, Q, alpha_P, alpha_D, alpha_Q)
        step = (
            np.linalg.norm(P - P_old, ord="fro")
            + np.linalg.norm(D - D_old, ord="fro")
            + np.linalg.norm(Q - Q_old, ord="fro")
        ) / max(1.0, np.linalg.norm(P_old, ord="fro") + np.linalg.norm(D_old, ord="fro") + np.linalg.norm(Q_old, ord="fro"))
        decrease = (obj_prev - obj) / max(1.0, abs(obj_prev))
        history.append({
            "sweep": sweep,
            "objective": obj,
            "relative_error_centered": rel_fro_error(Ac, Ahat_c),
            "relative_step": step,
            "relative_objective_decrease": decrease,
            "kappa_D": safe_cond(D),
        })
        if max(abs(decrease), step) <= tol:
            break
        obj_prev = obj

    elapsed = time.perf_counter() - start
    hist = pd.DataFrame(history)
    Ahat = mean + P @ D @ Q
    model = FittedLowRankModel(
        method_name,
        rr,
        mean,
        P @ D,
        Ahat,
        elapsed,
        safe_cond(D),
        len(history),
        storage_ratio_factorized(m, n, rr, "PDQ"),
        float(hist["relative_step"].iloc[-1]) if len(hist) else np.nan,
        float(hist["relative_objective_decrease"].iloc[-1]) if len(hist) else np.nan,
        {"P": P, "D": D, "Q": Q},
    )
    return model, hist


def fit_all_methods(A_train: np.ndarray, r: int, seed: int, cfg: dict) -> Tuple[List[FittedLowRankModel], Dict[str, pd.DataFrame]]:
    """Fit baselines, proposed variants, and ablations."""
    histories: Dict[str, pd.DataFrame] = {}
    methods: List[FittedLowRankModel] = [
        fit_mean_response(A_train),
        truncated_svd_fit(A_train, r),
        randomized_svd_fit(A_train, r, cfg.get("oversampling", 10), cfg.get("power_iter", 2), seed),
    ]

    model, hist = fit_pdq_model(
        A_train,
        r,
        alpha_P=cfg.get("alpha_P", 1e-4),
        alpha_D=cfg.get("alpha_D", 1e-4),
        alpha_Q=cfg.get("alpha_Q", 1e-4),
        max_sweeps=cfg.get("max_sweeps", 120),
        tol=cfg.get("tol", 1e-7),
        conditioned_core=False,
        kappa_core_max=cfg.get("kappa_core_max", 50.0),
        method_name="PDQ ridge",
        init="balanced",
        seed=seed,
    )
    methods.append(model)
    histories[model.method] = hist

    model, hist = fit_pdq_model(
        A_train,
        r,
        alpha_P=cfg.get("alpha_P", 1e-4),
        alpha_D=cfg.get("alpha_D", 1e-4),
        alpha_Q=cfg.get("alpha_Q", 1e-4),
        max_sweeps=cfg.get("max_sweeps", 120),
        tol=cfg.get("tol", 1e-7),
        conditioned_core=True,
        kappa_core_max=cfg.get("kappa_core_max", 50.0),
        method_name="PDQ conditioned core",
        init="balanced",
        seed=seed,
    )
    methods.append(model)
    histories[model.method] = hist

    model, hist = fit_pdq_model(
        A_train,
        r,
        alpha_P=cfg.get("weak_side_alpha", 1e-8),
        alpha_D=cfg.get("alpha_D", 1e-4),
        alpha_Q=cfg.get("weak_side_alpha", 1e-8),
        max_sweeps=cfg.get("max_sweeps", 120),
        tol=cfg.get("tol", 1e-7),
        conditioned_core=False,
        kappa_core_max=cfg.get("kappa_core_max", 50.0),
        method_name="PDQ weak-side-ridge ablation",
        init="balanced",
        seed=seed,
    )
    methods.append(model)
    histories[model.method] = hist

    model, hist = fit_pdq_model(
        A_train,
        r,
        alpha_P=cfg.get("alpha_P", 1e-4),
        alpha_D=cfg.get("alpha_D", 1e-4),
        alpha_Q=cfg.get("alpha_Q", 1e-4),
        max_sweeps=cfg.get("max_sweeps", 120),
        tol=cfg.get("tol", 1e-7),
        conditioned_core=True,
        kappa_core_max=cfg.get("kappa_core_max", 50.0),
        method_name="PDQ random-init ablation",
        init="random",
        seed=seed,
    )
    methods.append(model)
    histories[model.method] = hist
    return methods, histories
