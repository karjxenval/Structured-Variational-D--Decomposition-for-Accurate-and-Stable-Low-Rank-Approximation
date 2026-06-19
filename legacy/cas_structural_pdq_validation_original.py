#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Computers & Structures validation suite for core-conditioned PDQ surrogates.

This script adds the structural validation that the editor asked for:

1. A visible physical finite-element structure, not only an anonymous sparse matrix.
2. Physical response quantities: displacements, maximum displacement, strain energy,
   energy-norm error, and equilibrium residual.
3. Sparse-sensor reconstruction, which is the practical digital-twin setting.
4. Fair baselines: truncated SVD, randomized SVD, PDQ ridge, PDQ conditioned core,
   and a weak-side-ridge PDQ ablation.
5. Cost and deployment evidence: offline fitting cost, online reconstruction cost,
   storage ratio, sensor reduction, and speed-up against repeated FE solves.
6. Optional large-scale SuiteSparse validation using HB/bcsstk18 as a scalability check.

Dependencies
------------
    numpy scipy pandas matplotlib

Run
---
    python cas_structural_pdq_validation.py

Main outputs
------------
    results_cas_pdq_validation_YYYYMMDD_HHMMSS/
        config.json
        physical_model_summary.csv
        physical_all_runs.csv
        physical_aggregate.csv
        physical_main_table.csv
        physical_main_table.tex
        physical_full_snapshot_table.tex
        policy_summary.csv
        policy_summary.txt
        figures/physical_structure.png
        figures/sensor_recovery_error.png
        figures/energy_error.png
        figures/runtime_tradeoff.png
        figures/core_conditioning.png

If RUN_SUITE_SPARSE_VALIDATION is True, the folder also contains a large
SuiteSparse structural-matrix benchmark.
"""

from __future__ import annotations

import json
import time
import tarfile
import platform
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import sparse
from scipy.io import mmread
from scipy.linalg import solve
from scipy.sparse.linalg import splu


# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    "out_root": "results_cas_pdq_validation",

    "physical_model": {
        "name": "Two-dimensional steel truss bridge segment",
        "span_m": 30.0,
        "height_m": 4.0,
        "n_panels": 10,
        "E_pa": 200e9,
        "area_chord_m2": 4.0e-3,
        "area_vertical_m2": 2.5e-3,
        "area_diagonal_m2": 2.0e-3,
        "rho_kg_m3": 7850.0,
    },

    "physical_experiment": {
        "seeds": [13, 29, 47],
        "n_train": 80,
        "n_test": 40,
        "ranks": [4, 8, 12],
        "sensor_counts": [6, 10, 16],
        "training_noise_levels": [0.00, 0.01, 0.03],
        "sensor_noise_levels": [0.00, 0.01, 0.03],
        "coefficient_ridge": 1e-8,
        "load_scale_n": 8.0e4,
    },

    "pdq": {
        "alpha_P": 1e-4,
        "alpha_D": 1e-4,
        "alpha_Q": 1e-4,
        "weak_side_alpha": 1e-8,
        "max_sweeps": 120,
        "tol": 1e-7,
        "kappa_core_max": 50.0,
    },

    "randomized_svd": {
        "oversampling": 10,
        "power_iter": 2,
    },

    "run_suitesparse_validation": True,
    "suitesparse": {
        "group": "HB",
        "name": "bcsstk18",
        "url": "https://sparse.tamu.edu/MM/HB/bcsstk18.tar.gz",
        "description": "Structural stiffness matrix, R.E. Ginna Nuclear Power Station",
        "seeds": [13, 29, 47],
        "ranks": [10, 20, 30],
        "noise_levels": [0.00, 0.02, 0.05, 0.10],
        "n_scenarios": 120,
        "n_base_loads": 16,
        "load_sparsity": 45,
        "solve_shift_scale": 1e-10,
    },
}


# =============================================================================
# Utilities
# =============================================================================

def make_output_dir(out_root: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"{out_root}_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "figures").mkdir(exist_ok=True)
    (out_dir / "histories").mkdir(exist_ok=True)
    (out_dir / "data").mkdir(exist_ok=True)
    return out_dir


def save_config(config: dict, out_dir: Path) -> None:
    payload = dict(config)
    payload["system"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def rel_fro_error(A: np.ndarray, Ahat: np.ndarray) -> float:
    denom = np.linalg.norm(A, ord="fro")
    if denom <= 0:
        return np.nan
    return float(np.linalg.norm(A - Ahat, ord="fro") / denom)


def safe_cond(D: np.ndarray) -> float:
    try:
        s = np.linalg.svd(D, compute_uv=False)
        if len(s) == 0 or s[-1] <= 1e-14:
            return float("inf")
        return float(s[0] / s[-1])
    except np.linalg.LinAlgError:
        return float("inf")


def confidence_half_width(values: pd.Series) -> float:
    vals = values.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(vals) <= 1:
        return 0.0
    return float(1.96 * vals.std(ddof=1) / np.sqrt(len(vals)))


def storage_ratio_factorized(m: int, n: int, r: int, method: str) -> float:
    if "SVD" in method:
        stored = m * r + r + r * n
    elif "PDQ" in method:
        stored = m * r + r * r + r * n
    else:
        stored = m * r + r * n
    return float(stored / (m * n))


# =============================================================================
# Physical finite-element truss benchmark
# =============================================================================

@dataclass
class TrussModel:
    nodes: np.ndarray
    elements: List[Tuple[int, int, float, str]]
    fixed_dofs: np.ndarray
    free_dofs: np.ndarray
    K_full: np.ndarray
    K_ff: np.ndarray
    metadata: Dict[str, float | int | str]


def truss_element_stiffness(xi: np.ndarray, xj: np.ndarray, E: float, A: float) -> np.ndarray:
    dx, dy = xj - xi
    L = float(np.sqrt(dx * dx + dy * dy))
    if L <= 0:
        raise ValueError("Zero-length truss element.")
    c, s = dx / L, dy / L
    return (E * A / L) * np.array([
        [ c*c,  c*s, -c*c, -c*s],
        [ c*s,  s*s, -c*s, -s*s],
        [-c*c, -c*s,  c*c,  c*s],
        [-c*s, -s*s,  c*s,  s*s],
    ])


def build_truss_bridge_model(cfg: dict) -> TrussModel:
    L = float(cfg["span_m"])
    H = float(cfg["height_m"])
    n_panels = int(cfg["n_panels"])
    E = float(cfg["E_pa"])
    A_chord = float(cfg["area_chord_m2"])
    A_vert = float(cfg["area_vertical_m2"])
    A_diag = float(cfg["area_diagonal_m2"])

    xs = np.linspace(0.0, L, n_panels + 1)
    bottom = np.column_stack([xs, np.zeros_like(xs)])
    top = np.column_stack([xs, H * np.ones_like(xs)])
    nodes = np.vstack([bottom, top])

    n_bottom = n_panels + 1
    n_nodes = 2 * n_bottom

    elements: List[Tuple[int, int, float, str]] = []
    for i in range(n_panels):
        elements.append((i, i + 1, A_chord, "bottom_chord"))
        elements.append((n_bottom + i, n_bottom + i + 1, A_chord, "top_chord"))
    for i in range(n_bottom):
        elements.append((i, n_bottom + i, A_vert, "vertical"))
    for i in range(n_panels):
        elements.append((i, n_bottom + i + 1, A_diag, "diagonal"))
        elements.append((n_bottom + i, i + 1, A_diag, "diagonal"))

    ndof = 2 * n_nodes
    K = np.zeros((ndof, ndof), dtype=float)
    for ni, nj, area, _kind in elements:
        ke = truss_element_stiffness(nodes[ni], nodes[nj], E, area)
        dofs = np.array([2 * ni, 2 * ni + 1, 2 * nj, 2 * nj + 1], dtype=int)
        K[np.ix_(dofs, dofs)] += ke

    left = 0
    right = n_panels
    fixed = np.array([2 * left, 2 * left + 1, 2 * right + 1], dtype=int)
    all_dofs = np.arange(ndof, dtype=int)
    free = np.setdiff1d(all_dofs, fixed)
    K_ff = K[np.ix_(free, free)]

    eig_min = float(np.linalg.eigvalsh(K_ff).min())
    if eig_min <= 0:
        raise RuntimeError(f"Free stiffness matrix is not positive definite. min eigenvalue={eig_min:.3e}")

    metadata = {
        "name": cfg["name"],
        "span_m": L,
        "height_m": H,
        "n_panels": n_panels,
        "n_nodes": n_nodes,
        "n_elements": len(elements),
        "ndof_total": ndof,
        "ndof_free": len(free),
        "E_pa": E,
        "area_chord_m2": A_chord,
        "area_vertical_m2": A_vert,
        "area_diagonal_m2": A_diag,
        "min_eigenvalue_Kff": eig_min,
    }
    return TrussModel(nodes, elements, fixed, free, K, K_ff, metadata)


def plot_truss_model(model: TrussModel, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.2))
    colors = {"bottom_chord": "black", "top_chord": "black", "vertical": "gray", "diagonal": "gray"}
    for ni, nj, _area, kind in model.elements:
        xi, xj = model.nodes[ni], model.nodes[nj]
        ax.plot([xi[0], xj[0]], [xi[1], xj[1]], linewidth=1.5, color=colors.get(kind, "black"))
    ax.scatter(model.nodes[:, 0], model.nodes[:, 1], s=18)

    bottom_nodes = np.where(np.isclose(model.nodes[:, 1], 0.0))[0]
    left = bottom_nodes[np.argmin(model.nodes[bottom_nodes, 0])]
    right = bottom_nodes[np.argmax(model.nodes[bottom_nodes, 0])]
    ax.scatter([model.nodes[left, 0]], [model.nodes[left, 1]], marker="^", s=120, label="Pinned support")
    ax.scatter([model.nodes[right, 0]], [model.nodes[right, 1]], marker="s", s=90, label="Roller support")

    top_nodes = np.where(model.nodes[:, 1] > 0.0)[0]
    for node in top_nodes[1:-1:2]:
        x, y = model.nodes[node]
        ax.arrow(x, y + 0.8, 0, -0.55, head_width=0.25, head_length=0.15, length_includes_head=True)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x coordinate (m)")
    ax.set_ylabel("y coordinate (m)")
    ax.set_title("Physical structural benchmark: steel truss bridge segment")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "physical_structure.png", dpi=300)
    plt.close(fig)


def generate_structural_loads(model: TrussModel, n_cases: int, seed: int, load_scale: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ndof = model.K_full.shape[0]
    F = np.zeros((ndof, n_cases), dtype=float)
    nodes = model.nodes
    bottom_nodes = np.where(np.isclose(nodes[:, 1], 0.0))[0]
    top_nodes = np.where(nodes[:, 1] > 0.0)[0]
    interior_bottom = bottom_nodes[1:-1]
    interior_top = top_nodes[1:-1]
    t = np.linspace(0.0, 1.0, n_cases)

    for j in range(n_cases):
        fj = np.zeros(ndof, dtype=float)
        deck_amp = load_scale * (0.8 + 0.25 * np.sin(2 * np.pi * t[j]) + 0.10 * rng.normal())
        moving_amp = load_scale * (1.0 + 0.20 * np.cos(2 * np.pi * t[j]) + 0.10 * rng.normal())
        wind_amp = 0.25 * load_scale * (0.5 + 0.5 * np.sin(2 * np.pi * (t[j] + 0.15)) + 0.10 * rng.normal())

        for node in interior_bottom:
            fj[2 * node + 1] += -deck_amp / max(1, len(interior_bottom))

        moving_index = int(np.round(t[j] * (len(interior_bottom) - 1)))
        moving_node = int(interior_bottom[moving_index])
        fj[2 * moving_node + 1] += -moving_amp

        for node in interior_top:
            fj[2 * node] += wind_amp / max(1, len(interior_top))

        if j % 5 == 0:
            node = int(rng.choice(interior_top))
            fj[2 * node + 1] += -0.35 * load_scale * (1.0 + 0.10 * rng.normal())

        F[:, j] = fj
    return F


def solve_structural_responses(model: TrussModel, F_full: np.ndarray) -> Tuple[np.ndarray, float]:
    free = model.free_dofs
    K_ff = model.K_ff
    F_free = F_full[free, :]
    start = time.perf_counter()
    U_free = np.linalg.solve(K_ff, F_free)
    solve_time = time.perf_counter() - start
    U_full = np.zeros_like(F_full, dtype=float)
    U_full[free, :] = U_free
    return U_full, solve_time


def add_displacement_noise(U: np.ndarray, noise_level: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(1000 + seed)
    rms = np.linalg.norm(U, ord="fro") / np.sqrt(U.size)
    if rms <= 0:
        return U.copy()
    return U + noise_level * rms * rng.normal(size=U.shape)


# =============================================================================
# Low-rank baselines and PDQ
# =============================================================================

@dataclass
class FittedModel:
    method: str
    rank: int
    basis: np.ndarray
    Ahat_train: np.ndarray
    time_sec: float
    kappa_D: float
    sweeps: int
    storage_ratio: float
    final_step: float
    final_obj_decrease: float
    details: Dict[str, np.ndarray]


def truncated_svd_fit(A: np.ndarray, r: int) -> FittedModel:
    m, n = A.shape
    start = time.perf_counter()
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    B = U[:, :r]
    Ahat = (U[:, :r] * s[:r]) @ Vt[:r, :]
    elapsed = time.perf_counter() - start
    return FittedModel("Truncated SVD", r, B, Ahat, elapsed, np.nan, 0,
                       storage_ratio_factorized(m, n, r, "SVD"), np.nan, np.nan,
                       {"singular_values": s[:r], "Vt": Vt[:r, :]})


def randomized_svd_fit(A: np.ndarray, r: int, oversampling: int, n_iter: int, seed: int) -> FittedModel:
    rng = np.random.default_rng(2000 + seed)
    m, n = A.shape
    ell = min(n, r + oversampling)
    start = time.perf_counter()
    Omega = rng.normal(size=(n, ell))
    Y = A @ Omega
    for _ in range(n_iter):
        Y = A @ (A.T @ Y)
    Qbasis, _ = np.linalg.qr(Y, mode="reduced")
    Bsmall = Qbasis.T @ A
    Ub, s, Vt = np.linalg.svd(Bsmall, full_matrices=False)
    U = Qbasis @ Ub
    B = U[:, :r]
    Ahat = (U[:, :r] * s[:r]) @ Vt[:r, :]
    elapsed = time.perf_counter() - start
    return FittedModel("Randomized SVD", r, B, Ahat, elapsed, np.nan, 0,
                       storage_ratio_factorized(m, n, r, "SVD"), np.nan, np.nan,
                       {"singular_values": s[:r], "Vt": Vt[:r, :]})


def pdq_objective(A: np.ndarray, P: np.ndarray, D: np.ndarray, Q: np.ndarray,
                  alpha_P: float, alpha_D: float, alpha_Q: float) -> float:
    R = A - P @ D @ Q
    return float(0.5 * np.linalg.norm(R, ord="fro") ** 2
                 + 0.5 * alpha_P * np.linalg.norm(P, ord="fro") ** 2
                 + 0.5 * alpha_D * np.linalg.norm(D, ord="fro") ** 2
                 + 0.5 * alpha_Q * np.linalg.norm(Q, ord="fro") ** 2)


def init_pdq_balanced(A: np.ndarray, r: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    U = U[:, :r]
    s = np.maximum(s[:r], 1e-14)
    Vt = Vt[:r, :]
    root = np.sqrt(s)
    P = U * root
    D = np.eye(r)
    Q = root[:, None] * Vt
    return P, D, Q


def solve_D_update(P: np.ndarray, Q: np.ndarray, A: np.ndarray, alpha_D: float) -> np.ndarray:
    r = P.shape[1]
    Gp = P.T @ P
    Gq = Q @ Q.T
    C = P.T @ A @ Q.T
    L = np.kron(Gq.T, Gp) + alpha_D * np.eye(r * r)
    d = solve(L, C.reshape(-1, order="F"), assume_a="pos")
    return d.reshape((r, r), order="F")


def clip_core_condition(D: np.ndarray, kappa_max: float) -> np.ndarray:
    U, s, Vt = np.linalg.svd(D, full_matrices=False)
    smax = max(float(s[0]), 1e-14)
    smin = smax / kappa_max
    s_clipped = np.maximum(s, smin)
    return U @ np.diag(s_clipped) @ Vt


def fit_pdq_model(A: np.ndarray, r: int, alpha_P: float, alpha_D: float, alpha_Q: float,
                  max_sweeps: int, tol: float, conditioned_core: bool,
                  kappa_core_max: float, method_name: str) -> Tuple[FittedModel, pd.DataFrame]:
    m, n = A.shape
    P, D, Q = init_pdq_balanced(A, r)
    obj_prev = pdq_objective(A, P, D, Q, alpha_P, alpha_D, alpha_Q)
    history = []
    start = time.perf_counter()

    for sweep in range(1, max_sweeps + 1):
        P_old, D_old, Q_old = P.copy(), D.copy(), Q.copy()

        HP = D @ Q @ Q.T @ D.T + alpha_P * np.eye(r)
        P = A @ Q.T @ D.T @ np.linalg.inv(HP)

        D_candidate = solve_D_update(P, Q, A, alpha_D)
        if conditioned_core:
            D_projected = clip_core_condition(D_candidate, kappa_core_max)
            obj_before_D = pdq_objective(A, P, D_old, Q, alpha_P, alpha_D, alpha_Q)
            lam = 1.0
            accepted = False
            for _ in range(25):
                D_try = (1.0 - lam) * D_old + lam * D_projected
                obj_try = pdq_objective(A, P, D_try, Q, alpha_P, alpha_D, alpha_Q)
                if obj_try <= obj_before_D or lam < 1e-8:
                    D = D_try
                    accepted = True
                    break
                lam *= 0.5
            if not accepted:
                D = D_old
        else:
            D = D_candidate

        HQ = D.T @ P.T @ P @ D + alpha_Q * np.eye(r)
        RHS = D.T @ P.T @ A
        Q = np.linalg.solve(HQ, RHS)
        Ahat = P @ D @ Q
        obj = pdq_objective(A, P, D, Q, alpha_P, alpha_D, alpha_Q)

        step = (np.linalg.norm(P - P_old, ord="fro")
                + np.linalg.norm(D - D_old, ord="fro")
                + np.linalg.norm(Q - Q_old, ord="fro")) / max(
                    1.0,
                    np.linalg.norm(P_old, ord="fro")
                    + np.linalg.norm(D_old, ord="fro")
                    + np.linalg.norm(Q_old, ord="fro"),
                )
        decrease = (obj_prev - obj) / max(1.0, abs(obj_prev))
        history.append({
            "sweep": sweep,
            "objective": obj,
            "relative_error": rel_fro_error(A, Ahat),
            "relative_step": step,
            "relative_objective_decrease": decrease,
            "kappa_D": safe_cond(D),
        })
        if max(abs(decrease), step) <= tol:
            break
        obj_prev = obj

    elapsed = time.perf_counter() - start
    hist = pd.DataFrame(history)
    Ahat = P @ D @ Q
    basis = P @ D
    model = FittedModel(method_name, r, basis, Ahat, elapsed, safe_cond(D), len(history),
                        storage_ratio_factorized(m, n, r, "PDQ"),
                        float(hist["relative_step"].iloc[-1]) if len(hist) else np.nan,
                        float(hist["relative_objective_decrease"].iloc[-1]) if len(hist) else np.nan,
                        {"P": P, "D": D, "Q": Q})
    return model, hist


def fit_all_methods(A_train_noisy: np.ndarray, r: int, seed: int, cfg: dict) -> Tuple[List[FittedModel], Dict[str, pd.DataFrame]]:
    pdq_cfg = cfg["pdq"]
    rsvd_cfg = cfg["randomized_svd"]
    models: List[FittedModel] = []
    histories: Dict[str, pd.DataFrame] = {}

    models.append(truncated_svd_fit(A_train_noisy, r))
    models.append(randomized_svd_fit(A_train_noisy, r, rsvd_cfg["oversampling"], rsvd_cfg["power_iter"], seed))

    model, hist = fit_pdq_model(A_train_noisy, r, pdq_cfg["alpha_P"], pdq_cfg["alpha_D"], pdq_cfg["alpha_Q"],
                                pdq_cfg["max_sweeps"], pdq_cfg["tol"], False, pdq_cfg["kappa_core_max"], "PDQ ridge")
    models.append(model)
    histories[model.method] = hist

    model, hist = fit_pdq_model(A_train_noisy, r, pdq_cfg["alpha_P"], pdq_cfg["alpha_D"], pdq_cfg["alpha_Q"],
                                pdq_cfg["max_sweeps"], pdq_cfg["tol"], True, pdq_cfg["kappa_core_max"], "PDQ conditioned core")
    models.append(model)
    histories[model.method] = hist

    model, hist = fit_pdq_model(A_train_noisy, r, pdq_cfg["weak_side_alpha"], pdq_cfg["alpha_D"], pdq_cfg["weak_side_alpha"],
                                pdq_cfg["max_sweeps"], pdq_cfg["tol"], False, pdq_cfg["kappa_core_max"], "PDQ weak-side-ridge ablation")
    models.append(model)
    histories[model.method] = hist
    return models, histories


# =============================================================================
# Sensor reconstruction and physical metrics
# =============================================================================

def select_sensor_rows(A_train: np.ndarray, sensor_count: int) -> np.ndarray:
    row_var = np.var(A_train, axis=1)
    order = np.argsort(row_var)[::-1]
    return np.sort(order[:sensor_count])


def reconstruct_from_sensors(basis: np.ndarray, sensor_rows: np.ndarray, y: np.ndarray, coeff_ridge: float) -> np.ndarray:
    B_s = basis[sensor_rows, :]
    lhs = B_s.T @ B_s + coeff_ridge * np.eye(basis.shape[1])
    rhs = B_s.T @ y
    q = np.linalg.solve(lhs, rhs)
    return basis @ q


def physical_metrics(model: TrussModel, u_true: np.ndarray, u_hat: np.ndarray, f_full: np.ndarray) -> Dict[str, float]:
    free = model.free_dofs
    K_full = model.K_full
    K_ff = model.K_ff

    disp_err = float(np.linalg.norm(u_true - u_hat) / max(np.linalg.norm(u_true), 1e-30))
    max_true = float(np.max(np.abs(u_true)))
    max_hat = float(np.max(np.abs(u_hat)))
    max_disp_err = float(abs(max_true - max_hat) / max(max_true, 1e-30))

    energy_true = float(0.5 * u_true.T @ K_full @ u_true)
    energy_hat = float(0.5 * u_hat.T @ K_full @ u_hat)
    energy_err = float(abs(energy_true - energy_hat) / max(abs(energy_true), 1e-30))

    e = u_true[free] - u_hat[free]
    u_free = u_true[free]
    energy_norm_num = float(e.T @ K_ff @ e)
    energy_norm_den = float(u_free.T @ K_ff @ u_free)
    energy_norm_err = float(np.sqrt(max(0.0, energy_norm_num) / max(energy_norm_den, 1e-30)))

    f_free = f_full[free]
    residual = float(np.linalg.norm(K_ff @ u_hat[free] - f_free) / max(np.linalg.norm(f_free), 1e-30))

    return {
        "rel_displacement_error": disp_err,
        "rel_max_displacement_error": max_disp_err,
        "rel_strain_energy_error": energy_err,
        "rel_energy_norm_error": energy_norm_err,
        "rel_equilibrium_residual": residual,
        "true_max_displacement_m": max_true,
        "estimated_max_displacement_m": max_hat,
        "true_strain_energy_j": energy_true,
        "estimated_strain_energy_j": energy_hat,
    }


def evaluate_sparse_sensor_recovery(model: TrussModel, models: List[FittedModel], U_train_clean: np.ndarray,
                                    U_test_clean: np.ndarray, U_test_sensor_noisy: np.ndarray, F_test: np.ndarray,
                                    sensor_count: int, coeff_ridge: float, fe_online_time_per_case: float) -> List[dict]:
    rows = []
    sensor_rows = select_sensor_rows(U_train_clean, sensor_count)
    for fitted in models:
        start = time.perf_counter()
        metric_rows = []
        for j in range(U_test_clean.shape[1]):
            y = U_test_sensor_noisy[sensor_rows, j]
            u_hat = reconstruct_from_sensors(fitted.basis, sensor_rows, y, coeff_ridge)
            metric_rows.append(physical_metrics(model, U_test_clean[:, j], u_hat, F_test[:, j]))
        online_total = time.perf_counter() - start
        online_time_per_case = online_total / max(1, U_test_clean.shape[1])
        df = pd.DataFrame(metric_rows)
        rows.append({
            "method": fitted.method,
            "rank": fitted.rank,
            "sensor_count": sensor_count,
            "sensor_fraction": sensor_count / U_test_clean.shape[0],
            "mean_rel_displacement_error": df["rel_displacement_error"].mean(),
            "mean_rel_max_displacement_error": df["rel_max_displacement_error"].mean(),
            "mean_rel_strain_energy_error": df["rel_strain_energy_error"].mean(),
            "mean_rel_energy_norm_error": df["rel_energy_norm_error"].mean(),
            "mean_rel_equilibrium_residual": df["rel_equilibrium_residual"].mean(),
            "median_true_max_displacement_m": df["true_max_displacement_m"].median(),
            "median_true_strain_energy_j": df["true_strain_energy_j"].median(),
            "offline_fit_time_sec": fitted.time_sec,
            "online_time_per_case_sec": online_time_per_case,
            "fe_online_time_per_case_sec": fe_online_time_per_case,
            "online_speedup_vs_fe": fe_online_time_per_case / max(online_time_per_case, 1e-30),
            "storage_ratio": fitted.storage_ratio,
            "kappa_D": fitted.kappa_D,
            "sweeps": fitted.sweeps,
            "final_step": fitted.final_step,
            "final_obj_decrease": fitted.final_obj_decrease,
        })
    return rows


def evaluate_full_snapshot_recovery(U_train_clean: np.ndarray, U_train_noisy: np.ndarray, models: List[FittedModel]) -> List[dict]:
    rows = []
    for fitted in models:
        rows.append({
            "method": fitted.method,
            "rank": fitted.rank,
            "train_clean_response_error": rel_fro_error(U_train_clean, fitted.Ahat_train),
            "train_noisy_response_error": rel_fro_error(U_train_noisy, fitted.Ahat_train),
            "offline_fit_time_sec": fitted.time_sec,
            "storage_ratio": fitted.storage_ratio,
            "kappa_D": fitted.kappa_D,
            "sweeps": fitted.sweeps,
        })
    return rows


# =============================================================================
# Aggregation, tables, figures
# =============================================================================

def aggregate_physical_results(all_runs: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["training_noise_level", "sensor_noise_level", "rank", "sensor_count", "method"]
    metric_cols = [
        "mean_rel_displacement_error",
        "mean_rel_max_displacement_error",
        "mean_rel_strain_energy_error",
        "mean_rel_energy_norm_error",
        "mean_rel_equilibrium_residual",
        "offline_fit_time_sec",
        "online_time_per_case_sec",
        "online_speedup_vs_fe",
        "storage_ratio",
        "kappa_D",
        "sweeps",
        "sensor_fraction",
    ]
    rows = []
    for keys, g in all_runs.groupby(group_cols):
        rec = dict(zip(group_cols, keys))
        rec["n_cases"] = len(g)
        for col in metric_cols:
            vals = g[col].replace([np.inf, -np.inf], np.nan)
            rec[f"{col}_mean"] = vals.mean()
            rec[f"{col}_ci95"] = confidence_half_width(vals)
            rec[f"{col}_median"] = vals.median()
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(group_cols)


def aggregate_full_snapshot(full_runs: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["noise_level", "rank", "method"]
    rows = []
    for keys, g in full_runs.groupby(group_cols):
        rec = dict(zip(group_cols, keys))
        rec["n_cases"] = len(g)
        for col in ["train_clean_response_error", "train_noisy_response_error", "offline_fit_time_sec", "storage_ratio", "kappa_D"]:
            vals = g[col].replace([np.inf, -np.inf], np.nan)
            rec[f"{col}_mean"] = vals.mean()
            rec[f"{col}_ci95"] = confidence_half_width(vals)
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(group_cols)


def make_physical_tables(aggregate: pd.DataFrame, full_aggregate: pd.DataFrame, out_dir: Path) -> None:
    target = aggregate.copy()
    target["score"] = ((target["rank"] - 8).abs()
                       + (target["sensor_count"] - 10).abs()
                       + 100 * (target["training_noise_level"] - 0.01).abs()
                       + 100 * (target["sensor_noise_level"] - 0.01).abs())
    best_setting = target.loc[target["score"].idxmin()]
    subset = aggregate[
        (aggregate["rank"] == best_setting["rank"])
        & (aggregate["sensor_count"] == best_setting["sensor_count"])
        & (aggregate["training_noise_level"] == best_setting["training_noise_level"])
        & (aggregate["sensor_noise_level"] == best_setting["sensor_noise_level"])
    ].copy()
    subset = subset[[
        "method", "rank", "sensor_count", "training_noise_level", "sensor_noise_level",
        "mean_rel_displacement_error_mean", "mean_rel_energy_norm_error_mean",
        "mean_rel_strain_energy_error_mean", "mean_rel_equilibrium_residual_mean",
        "online_speedup_vs_fe_mean", "storage_ratio_mean", "kappa_D_mean",
    ]]
    subset.to_csv(out_dir / "physical_main_table.csv", index=False)
    tex = subset.to_latex(
        index=False,
        float_format=lambda x: f"{x:.3e}" if np.isfinite(x) else "--",
        caption=("Physical structural benchmark under sparse-sensor reconstruction. "
                 "Errors are computed against clean finite-element displacements and energy quantities."),
        label="tab:physical_structural_sparse_sensor",
    )
    (out_dir / "physical_main_table.tex").write_text(tex, encoding="utf-8")
    tex2 = full_aggregate.to_latex(
        index=False,
        float_format=lambda x: f"{x:.3e}" if np.isfinite(x) else "--",
        caption=("Full-snapshot training reconstruction on the physical structural benchmark. "
                 "Clean-response error measures denoising quality relative to the noiseless finite-element response matrix."),
        label="tab:physical_full_snapshot",
    )
    (out_dir / "physical_full_snapshot_table.tex").write_text(tex2, encoding="utf-8")


def make_policy_summary(aggregate: pd.DataFrame, out_dir: Path) -> None:
    subset = aggregate[(aggregate["rank"] == 8)
                       & (np.isclose(aggregate["training_noise_level"], 0.01))
                       & (np.isclose(aggregate["sensor_noise_level"], 0.01))]
    if subset.empty:
        subset = aggregate.copy()
    main = subset[subset["method"].isin(["Truncated SVD", "Randomized SVD", "PDQ ridge", "PDQ conditioned core"])].copy()
    if main.empty:
        main = subset.copy()
    best_disp = main.loc[main["mean_rel_displacement_error_mean"].idxmin()]
    best_energy = main.loc[main["mean_rel_energy_norm_error_mean"].idxmin()]
    pdq_only = main[main["method"].str.contains("PDQ", regex=False)].copy()
    if not pdq_only.empty:
        best_stability = pdq_only.loc[pdq_only["kappa_D_mean"].replace([np.inf, -np.inf], np.nan).fillna(np.inf).idxmin()]
    else:
        best_stability = best_disp

    rows = [{
        "message": "best_mean_displacement_error_method",
        "method": best_disp["method"],
        "value": best_disp["mean_rel_displacement_error_mean"],
        "rank": best_disp["rank"],
        "sensor_count": best_disp["sensor_count"],
    }, {
        "message": "best_mean_energy_norm_error_method",
        "method": best_energy["method"],
        "value": best_energy["mean_rel_energy_norm_error_mean"],
        "rank": best_energy["rank"],
        "sensor_count": best_energy["sensor_count"],
    }, {
        "message": "best_reported_core_conditioning_among_pdq_methods",
        "method": best_stability["method"],
        "value": best_stability["kappa_D_mean"],
        "rank": best_stability["rank"],
        "sensor_count": best_stability["sensor_count"],
    }]
    pd.DataFrame(rows).to_csv(out_dir / "policy_summary.csv", index=False)

    text = []
    text.append("Decision-facing summary")
    text.append("======================")
    text.append("")
    text.append("The physical benchmark evaluates whether a reduced surrogate can reconstruct")
    text.append("full-field structural displacements from sparse sensor measurements while")
    text.append("also reporting stability diagnostics.")
    text.append("")
    text.append(f"Best mean displacement-error method: {best_disp['method']} ({best_disp['mean_rel_displacement_error_mean']:.3e}).")
    text.append(f"Best mean energy-norm-error method: {best_energy['method']} ({best_energy['mean_rel_energy_norm_error_mean']:.3e}).")
    if "PDQ" in str(best_stability["method"]):
        text.append(f"Best reported PDQ core conditioning: {best_stability['method']} (kappa(D)={best_stability['kappa_D_mean']:.3e}).")
    text.append("")
    text.append("Interpretation: SVD remains the reference for clean full-data compression. PDQ")
    text.append("should be judged by whether response recovery and core-conditioning diagnostics")
    text.append("justify its additional offline fitting cost.")
    (out_dir / "policy_summary.txt").write_text("\n".join(text), encoding="utf-8")


def make_physical_figures(aggregate: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    plot_df = aggregate[(np.isclose(aggregate["training_noise_level"], 0.01))
                        & (np.isclose(aggregate["sensor_noise_level"], 0.01))
                        & (aggregate["rank"] == 8)].copy()
    if plot_df.empty:
        plot_df = aggregate.copy()
    main_methods = ["Truncated SVD", "Randomized SVD", "PDQ ridge", "PDQ conditioned core"]
    plot_df = plot_df[plot_df["method"].isin(main_methods)]

    plt.figure(figsize=(8.5, 5))
    for method, g in plot_df.groupby("method"):
        g = g.sort_values("sensor_count")
        plt.plot(g["sensor_count"], g["mean_rel_displacement_error_mean"], marker="o", label=method)
    plt.xlabel("Number of measured displacement DOFs")
    plt.ylabel("Mean relative full-field displacement error")
    plt.title("Sparse-sensor full-field reconstruction")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "sensor_recovery_error.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8.5, 5))
    for method, g in plot_df.groupby("method"):
        g = g.sort_values("sensor_count")
        plt.plot(g["sensor_count"], g["mean_rel_energy_norm_error_mean"], marker="o", label=method)
    plt.xlabel("Number of measured displacement DOFs")
    plt.ylabel("Mean relative energy-norm error")
    plt.title("Physical error under sparse sensing")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "energy_error.png", dpi=300)
    plt.close()

    rt = plot_df[plot_df["sensor_count"] == plot_df["sensor_count"].median()].copy()
    if not rt.empty:
        plt.figure(figsize=(8.5, 5))
        plt.bar(rt["method"], rt["offline_fit_time_sec_mean"])
        plt.yscale("log")
        plt.ylabel("Offline fit time (s), log scale")
        plt.xticks(rotation=25, ha="right")
        plt.title("Offline surrogate fitting cost")
        plt.tight_layout()
        plt.savefig(fig_dir / "runtime_tradeoff.png", dpi=300)
        plt.close()

    kdf = plot_df[plot_df["method"].str.contains("PDQ", regex=False)].copy()
    if not kdf.empty:
        kdf = kdf.groupby("method", as_index=False)["kappa_D_mean"].median()
        plt.figure(figsize=(8.5, 5))
        plt.bar(kdf["method"], kdf["kappa_D_mean"])
        plt.yscale("log")
        plt.ylabel(r"Median learned core condition number $\kappa(D)$")
        plt.xticks(rotation=25, ha="right")
        plt.title("Core-conditioning diagnostic")
        plt.tight_layout()
        plt.savefig(fig_dir / "core_conditioning.png", dpi=300)
        plt.close()


# =============================================================================
# Physical benchmark driver
# =============================================================================

def run_physical_benchmark(config: dict, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 78)
    print("PHYSICAL STRUCTURAL BENCHMARK")
    print("=" * 78)
    model = build_truss_bridge_model(config["physical_model"])
    plot_truss_model(model, out_dir)
    pd.DataFrame([model.metadata]).to_csv(out_dir / "physical_model_summary.csv", index=False)

    exp = config["physical_experiment"]
    all_rows = []
    full_rows = []
    for seed in exp["seeds"]:
        n_total = exp["n_train"] + exp["n_test"]
        F_full = generate_structural_loads(model, n_total, seed, exp["load_scale_n"])
        U_clean, fe_total_solve_time = solve_structural_responses(model, F_full)
        fe_online_time_per_case = fe_total_solve_time / max(1, n_total)

        U_train_clean = U_clean[:, :exp["n_train"]]
        U_test_clean = U_clean[:, exp["n_train"]:]
        F_test = F_full[:, exp["n_train"]:]

        for training_noise in exp["training_noise_levels"]:
            U_train_noisy = add_displacement_noise(U_train_clean, training_noise, seed)
            for rank in exp["ranks"]:
                models, histories = fit_all_methods(U_train_noisy, rank, seed, config)
                for method_name, hist in histories.items():
                    safe_name = method_name.lower().replace(" ", "_").replace("-", "_")
                    hist.to_csv(out_dir / "histories" / f"physical_{safe_name}_seed{seed}_noise{training_noise}_rank{rank}.csv", index=False)

                for rec in evaluate_full_snapshot_recovery(U_train_clean, U_train_noisy, models):
                    rec.update({"seed": seed, "noise_level": training_noise, "fe_online_time_per_case_sec": fe_online_time_per_case})
                    full_rows.append(rec)

                for sensor_noise in exp["sensor_noise_levels"]:
                    U_test_sensor_noisy = add_displacement_noise(U_test_clean, sensor_noise, seed + 5000 + int(1000 * training_noise))
                    for sensor_count in exp["sensor_counts"]:
                        rows = evaluate_sparse_sensor_recovery(
                            model, models, U_train_clean, U_test_clean, U_test_sensor_noisy, F_test,
                            sensor_count, exp["coefficient_ridge"], fe_online_time_per_case,
                        )
                        for rec in rows:
                            rec.update({
                                "seed": seed,
                                "training_noise_level": training_noise,
                                "sensor_noise_level": sensor_noise,
                                "n_train": exp["n_train"],
                                "n_test": exp["n_test"],
                                "ndof_total": model.metadata["ndof_total"],
                                "ndof_free": model.metadata["ndof_free"],
                            })
                            all_rows.append(rec)

    all_runs = pd.DataFrame(all_rows)
    full_runs = pd.DataFrame(full_rows)
    all_runs.to_csv(out_dir / "physical_all_runs.csv", index=False)
    full_runs.to_csv(out_dir / "physical_full_snapshot_runs.csv", index=False)
    aggregate = aggregate_physical_results(all_runs)
    full_aggregate = aggregate_full_snapshot(full_runs)
    aggregate.to_csv(out_dir / "physical_aggregate.csv", index=False)
    full_aggregate.to_csv(out_dir / "physical_full_snapshot_aggregate.csv", index=False)
    make_physical_tables(aggregate, full_aggregate, out_dir)
    make_policy_summary(aggregate, out_dir)
    make_physical_figures(aggregate, out_dir)
    return all_runs, aggregate


# =============================================================================
# Optional SuiteSparse large structural benchmark
# =============================================================================

def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[INFO] Using existing download: {dest}")
        return
    print(f"[INFO] Downloading {url}")
    urllib.request.urlretrieve(url, dest)


def extract_matrix_market(tar_path: Path, extract_dir: Path) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    marker = extract_dir / ".extracted"
    if not marker.exists():
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(extract_dir)
        marker.write_text("done", encoding="utf-8")
    mtx_files = list(extract_dir.rglob("*.mtx"))
    if not mtx_files:
        raise FileNotFoundError(f"No .mtx file found in {extract_dir}")
    return mtx_files[0]


def load_suitesparse_matrix(ss_cfg: dict, out_dir: Path) -> sparse.csc_matrix:
    tar_path = out_dir / "data" / f"{ss_cfg['name']}.tar.gz"
    extract_dir = out_dir / "data" / ss_cfg["name"]
    download_file(ss_cfg["url"], tar_path)
    mtx_path = extract_matrix_market(tar_path, extract_dir)
    K = mmread(str(mtx_path)).tocsc().astype(float)
    return (0.5 * (K + K.T)).tocsc()


def make_suitesparse_loads(n_dof: int, n_scenarios: int, n_base_loads: int, load_sparsity: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.zeros((n_dof, n_base_loads))
    for k in range(n_base_loads):
        idx = rng.choice(n_dof, size=load_sparsity, replace=False)
        base[idx, k] = rng.normal(0.0, 1.0, size=load_sparsity)
    t = np.linspace(0.0, 1.0, n_scenarios)
    coeff = rng.normal(0.0, 1.0, size=(n_base_loads, n_scenarios))
    for k in range(n_base_loads):
        coeff[k, :] += 0.6 * np.sin(2 * np.pi * (k + 1) * t / max(2, n_base_loads))
        coeff[k, :] += 0.3 * np.cos(2 * np.pi * (k + 2) * t / max(3, n_base_loads))
    B = base @ coeff
    norm = np.linalg.norm(B, ord="fro")
    if norm > 0:
        B = B / norm * np.sqrt(n_scenarios)
    return B


def build_suitesparse_response(K: sparse.csc_matrix, ss_cfg: dict, seed: int) -> np.ndarray:
    n = K.shape[0]
    B = make_suitesparse_loads(n, ss_cfg["n_scenarios"], ss_cfg["n_base_loads"], ss_cfg["load_sparsity"], seed)
    diag_mean = float(np.mean(np.abs(K.diagonal())))
    shift = ss_cfg["solve_shift_scale"] * (diag_mean if diag_mean > 0 else 1.0)
    K_solve = (K + shift * sparse.eye(n, format="csc")).tocsc()
    lu = splu(K_solve)
    X = lu.solve(B)
    A0 = X - X.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(A0, ord="fro")
    if norm > 0:
        A0 = A0 / norm
    return A0


def aggregate_suitesparse_results(all_runs: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["noise_level", "rank", "method"]
    rows = []
    for keys, g in all_runs.groupby(group_cols):
        rec = dict(zip(group_cols, keys))
        rec["n_cases"] = len(g)
        for col in ["clean_error", "noisy_error", "time_sec", "storage_ratio", "kappa_D", "sweeps"]:
            vals = g[col].replace([np.inf, -np.inf], np.nan)
            rec[f"{col}_mean"] = vals.mean()
            rec[f"{col}_ci95"] = confidence_half_width(vals)
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(group_cols)


def make_suitesparse_figures(aggregate: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    main_methods = ["Truncated SVD", "Randomized SVD", "PDQ ridge", "PDQ conditioned core"]
    g = aggregate[aggregate["method"].isin(main_methods)].copy()
    if g.empty:
        return
    pivot = g.groupby(["noise_level", "method"], as_index=False)["clean_error_mean"].mean()
    pivot = pivot.pivot(index="noise_level", columns="method", values="clean_error_mean")
    plt.figure(figsize=(8.5, 5))
    for col in pivot.columns:
        plt.plot(pivot.index, pivot[col], marker="o", label=col)
    plt.xlabel("Noise level")
    plt.ylabel("Mean clean-response error")
    plt.title("Large sparse structural response benchmark")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "suitesparse_clean_error.png", dpi=300)
    plt.close()


def run_suitesparse_validation(config: dict, out_dir: Path) -> None:
    if not config.get("run_suitesparse_validation", False):
        return
    print("\n" + "=" * 78)
    print("LARGE SUITESPARSE STRUCTURAL BENCHMARK")
    print("=" * 78)
    ss_cfg = config["suitesparse"]
    K = load_suitesparse_matrix(ss_cfg, out_dir)
    pd.DataFrame([{
        "matrix_group": ss_cfg["group"],
        "matrix_name": ss_cfg["name"],
        "description": ss_cfg["description"],
        "K_rows": K.shape[0],
        "K_cols": K.shape[1],
        "K_nnz": K.nnz,
        "n_scenarios": ss_cfg["n_scenarios"],
        "n_base_loads": ss_cfg["n_base_loads"],
        "load_sparsity": ss_cfg["load_sparsity"],
    }]).to_csv(out_dir / "suitesparse_dataset_summary.csv", index=False)

    rows = []
    for seed in ss_cfg["seeds"]:
        A0 = build_suitesparse_response(K, ss_cfg, seed)
        for noise_level in ss_cfg["noise_levels"]:
            A = add_displacement_noise(A0, noise_level, seed + 7000)
            for rank in ss_cfg["ranks"]:
                models, histories = fit_all_methods(A, rank, seed, config)
                for method_name, hist in histories.items():
                    safe_name = method_name.lower().replace(" ", "_").replace("-", "_")
                    hist.to_csv(out_dir / "histories" / f"suitesparse_{safe_name}_seed{seed}_noise{noise_level}_rank{rank}.csv", index=False)
                for fitted in models:
                    rows.append({
                        "seed": seed,
                        "noise_level": noise_level,
                        "rank": rank,
                        "method": fitted.method,
                        "clean_error": rel_fro_error(A0, fitted.Ahat_train),
                        "noisy_error": rel_fro_error(A, fitted.Ahat_train),
                        "time_sec": fitted.time_sec,
                        "storage_ratio": fitted.storage_ratio,
                        "kappa_D": fitted.kappa_D,
                        "sweeps": fitted.sweeps,
                    })
    all_runs = pd.DataFrame(rows)
    all_runs.to_csv(out_dir / "suitesparse_all_runs.csv", index=False)
    aggregate = aggregate_suitesparse_results(all_runs)
    aggregate.to_csv(out_dir / "suitesparse_aggregate.csv", index=False)
    tex = aggregate.to_latex(index=False, float_format=lambda x: f"{x:.3e}" if np.isfinite(x) else "--",
                             caption="Large sparse structural-matrix validation using the SuiteSparse \\texttt{HB/bcsstk18} stiffness matrix.",
                             label="tab:suitesparse_large_validation")
    (out_dir / "suitesparse_main_table.tex").write_text(tex, encoding="utf-8")
    make_suitesparse_figures(aggregate, out_dir)


# =============================================================================
# Main
# =============================================================================

def run_all(config: dict) -> None:
    out_dir = make_output_dir(config["out_root"])
    save_config(config, out_dir)
    print("\n" + "=" * 78)
    print("COMPUTERS & STRUCTURES PDQ VALIDATION SUITE")
    print("=" * 78)
    print(f"[INFO] Output folder: {out_dir.resolve()}")
    run_physical_benchmark(config, out_dir)
    run_suitesparse_validation(config, out_dir)
    print("\n" + "=" * 78)
    print("DONE")
    print("=" * 78)
    print(f"Results saved in: {out_dir.resolve()}")
    print("\nMost important files:")
    for name in [
        "physical_model_summary.csv",
        "physical_all_runs.csv",
        "physical_aggregate.csv",
        "physical_main_table.csv",
        "physical_main_table.tex",
        "physical_full_snapshot_table.tex",
        "policy_summary.txt",
        "figures/physical_structure.png",
        "figures/sensor_recovery_error.png",
        "figures/energy_error.png",
        "figures/runtime_tradeoff.png",
        "figures/core_conditioning.png",
        "suitesparse_aggregate.csv",
        "suitesparse_main_table.tex",
    ]:
        path = out_dir / name
        if path.exists():
            print(f"  {path}")


if __name__ == "__main__":
    run_all(CONFIG)
