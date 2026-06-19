"""
Industrial-grade validation for core-conditioned PDQ factorization.

This script is designed for a serious scientific-computing/industry-facing paper.

It does NOT try to cherry-pick one nice run. Instead, it runs a pre-specified
benchmark grid on a real SuiteSparse structural stiffness matrix.

Data source
-----------
Default matrix:
    HB/bcsstk18
    Structural stiffness matrix, R.E. Ginna Nuclear Power Station.
    Source: SuiteSparse Matrix Collection.

Workflow
--------
1. Download real sparse structural matrix K from SuiteSparse.
2. Generate physically plausible load cases B.
3. Solve K X = B to obtain response snapshots.
4. Form response matrix A0 = X.
5. Add controlled sensor/simulation noise to obtain A.
6. Compare:
      - Truncated SVD
      - Randomized SVD
      - Proposed PDQ ridge
      - Proposed PDQ conditioned core
      - PDQ weak-side-ridge ablation
7. Repeat over several ranks, noise levels, and random seeds.
8. Save essential evidence tables and compact plots.

Main evidence
-------------
- Clean-response error:       ||A0 - Ahat||_F / ||A0||_F
- Noisy-response error:       ||A  - Ahat||_F / ||A||_F
- Denoising gain vs SVD:      relative improvement in clean-response error
- Core condition number:      kappa(D)
- Runtime
- Storage ratio
- Paired win rates over repeated runs

Outputs
-------
A new folder such as:

    results_industrial_pdq_v2_20260515_174200/

containing:

    config.json
    dataset_summary.csv
    all_runs.csv
    aggregate_table.csv
    paired_comparison_table.csv
    manuscript_table.tex
    summary_clean_error.png
    summary_runtime.png
    summary_kappa.png
"""

from __future__ import annotations

import json
import time
import tarfile
import platform
import urllib.request
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import sparse
from scipy.io import mmread
from scipy.linalg import solve
from scipy.sparse.linalg import splu


# ============================================================
# Configuration
# ============================================================

CONFIG = {
    # A fresh timestamped folder will be created inside this root.
    "out_root": "results_industrial_pdq_v2",

    # Real public structural matrix from SuiteSparse.
    "matrix": {
        "group": "HB",
        "name": "bcsstk18",
        "url": "https://sparse.tamu.edu/MM/HB/bcsstk18.tar.gz",
        "description": "Structural stiffness matrix, R.E. Ginna Nuclear Power Station",
    },

    # Benchmark grid.
    # Keep this moderate first. Increase later if your machine can handle it.
    "seeds": [13, 29, 47],
    "ranks": [10, 20, 30],
    "noise_levels": [0.00, 0.02, 0.05, 0.10],

    # Response-matrix construction.
    "n_scenarios": 120,
    "n_base_loads": 16,
    "load_sparsity": 45,

    # Solver safety for building response snapshots.
    # This is only for Kx=b construction, not for PDQ itself.
    "solve_shift_scale": 1e-10,

    # PDQ fitting.
    "alpha_P": 1e-4,
    "alpha_D": 1e-4,
    "alpha_Q": 1e-4,
    "max_sweeps": 120,
    "tol": 1e-7,

    # Conditioned-core variant.
    "kappa_core_max": 50.0,

    # Weak-side-ridge ablation. This is included to show why side-factor
    # regularization matters. It should NOT be presented as the main method.
    "weak_side_alpha": 1e-8,

    # Randomized SVD settings.
    "randomized_oversampling": 10,
    "randomized_power_iter": 2,
}


# ============================================================
# Basic utilities
# ============================================================

def make_output_dir(out_root: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"{out_root}_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "data").mkdir(exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    (out_dir / "histories").mkdir(exist_ok=True)
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


def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[INFO] Using existing file: {dest}")
        return
    print(f"[INFO] Downloading {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"[INFO] Saved: {dest}")


def extract_matrix_market(tar_path: Path, extract_dir: Path) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    marker = extract_dir / ".extracted"

    if not marker.exists():
        print(f"[INFO] Extracting {tar_path}")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(extract_dir)
        marker.write_text("done", encoding="utf-8")

    mtx_files = list(extract_dir.rglob("*.mtx"))
    if not mtx_files:
        raise FileNotFoundError(f"No Matrix Market file found in {extract_dir}")

    return mtx_files[0]


def load_suitesparse_matrix(config: dict, out_dir: Path) -> sparse.csc_matrix:
    matrix = config["matrix"]
    tar_path = out_dir / "data" / f"{matrix['name']}.tar.gz"
    extract_dir = out_dir / "data" / matrix["name"]

    download_file(matrix["url"], tar_path)
    mtx_path = extract_matrix_market(tar_path, extract_dir)

    print(f"[INFO] Reading matrix: {mtx_path}")
    K = mmread(str(mtx_path)).tocsc().astype(float)

    if K.shape[0] != K.shape[1]:
        raise ValueError("This validation expects a square stiffness/operator matrix.")

    # Symmetrize because structural stiffness matrices are symmetric, but
    # storage may contain only one triangle or minor numerical asymmetry.
    K = 0.5 * (K + K.T)
    K = K.tocsc()

    print(f"[INFO] Loaded K: shape={K.shape}, nnz={K.nnz}")
    return K


def rel_fro_error(A: np.ndarray, Ahat: np.ndarray) -> float:
    denom = np.linalg.norm(A, ord="fro")
    if denom == 0:
        return np.nan
    return float(np.linalg.norm(A - Ahat, ord="fro") / denom)


def storage_ratio_pdq(m: int, n: int, r: int) -> float:
    return float((m * r + r * r + r * n) / (m * n))


def safe_cond(D: np.ndarray) -> float:
    s = np.linalg.svd(D, compute_uv=False)
    if len(s) == 0 or s[-1] <= 1e-14:
        return float("inf")
    return float(s[0] / s[-1])


def confidence_half_width(values: pd.Series) -> float:
    vals = values.dropna().to_numpy(dtype=float)
    if len(vals) <= 1:
        return 0.0
    return float(1.96 * vals.std(ddof=1) / np.sqrt(len(vals)))


# ============================================================
# Response-matrix construction
# ============================================================

def add_diagonal_shift(K: sparse.csc_matrix, shift_scale: float) -> sparse.csc_matrix:
    diag_mean = float(np.mean(np.abs(K.diagonal())))
    if not np.isfinite(diag_mean) or diag_mean <= 0:
        diag_mean = 1.0
    shift = shift_scale * diag_mean
    print(f"[INFO] Diagonal shift for response solve: {shift:.3e}")
    return (K + shift * sparse.eye(K.shape[0], format="csc")).tocsc()


def make_structured_loads(
    n_dof: int,
    n_scenarios: int,
    n_base_loads: int,
    load_sparsity: int,
    seed: int,
) -> np.ndarray:
    """
    Generate load cases from localized base loads.

    This is not cartoon random noise. It mimics a common industrial workflow:
    many operating scenarios are combinations of a smaller number of localized
    forcing patterns.
    """
    rng = np.random.default_rng(seed)

    base = np.zeros((n_dof, n_base_loads))
    for k in range(n_base_loads):
        idx = rng.choice(n_dof, size=load_sparsity, replace=False)
        base[idx, k] = rng.normal(0.0, 1.0, size=load_sparsity)

    t = np.linspace(0.0, 1.0, n_scenarios)
    coeff = rng.normal(0.0, 1.0, size=(n_base_loads, n_scenarios))

    # Add smooth operating regimes so columns are not independent white noise.
    for k in range(n_base_loads):
        coeff[k, :] += 0.6 * np.sin(2 * np.pi * (k + 1) * t / max(2, n_base_loads))
        coeff[k, :] += 0.3 * np.cos(2 * np.pi * (k + 2) * t / max(3, n_base_loads))

    B = base @ coeff

    norm = np.linalg.norm(B, ord="fro")
    if norm > 0:
        B = B / norm * np.sqrt(n_scenarios)

    return B


def build_response_matrix(K: sparse.csc_matrix, config: dict, seed: int) -> np.ndarray:
    n = K.shape[0]

    B = make_structured_loads(
        n_dof=n,
        n_scenarios=config["n_scenarios"],
        n_base_loads=config["n_base_loads"],
        load_sparsity=config["load_sparsity"],
        seed=seed,
    )

    K_solve = add_diagonal_shift(K, config["solve_shift_scale"])

    start = time.perf_counter()
    lu = splu(K_solve)
    factor_time = time.perf_counter() - start

    start = time.perf_counter()
    X = lu.solve(B)
    solve_time = time.perf_counter() - start

    print(f"[INFO] Sparse LU time: {factor_time:.2f}s, multi-RHS solve time: {solve_time:.2f}s")

    A0 = X - X.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(A0, ord="fro")
    if norm > 0:
        A0 = A0 / norm

    return A0


def add_response_noise(A0: np.ndarray, noise_level: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(10_000 + seed)
    rms = np.linalg.norm(A0, ord="fro") / np.sqrt(A0.size)
    noise = noise_level * rms * rng.normal(size=A0.shape)
    return A0 + noise


# ============================================================
# Baselines
# ============================================================

def truncated_svd_approx(A: np.ndarray, r: int) -> tuple[np.ndarray, dict]:
    start = time.perf_counter()
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    Ahat = (U[:, :r] * s[:r]) @ Vt[:r, :]
    elapsed = time.perf_counter() - start

    return Ahat, {
        "time_sec": elapsed,
        "sweeps": 0,
        "kappa_D": np.nan,
        "final_step": np.nan,
        "final_obj_decrease": np.nan,
    }


def randomized_svd_approx(
    A: np.ndarray,
    r: int,
    oversampling: int,
    n_iter: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(20_000 + seed)
    m, n = A.shape
    ell = min(n, r + oversampling)

    start = time.perf_counter()

    Omega = rng.normal(size=(n, ell))
    Y = A @ Omega

    for _ in range(n_iter):
        Y = A @ (A.T @ Y)

    Qbasis, _ = np.linalg.qr(Y, mode="reduced")
    B = Qbasis.T @ A
    Ub, s, Vt = np.linalg.svd(B, full_matrices=False)
    U = Qbasis @ Ub

    Ahat = (U[:, :r] * s[:r]) @ Vt[:r, :]
    elapsed = time.perf_counter() - start

    return Ahat, {
        "time_sec": elapsed,
        "sweeps": 0,
        "kappa_D": np.nan,
        "final_step": np.nan,
        "final_obj_decrease": np.nan,
    }


# ============================================================
# PDQ fitting
# ============================================================

def pdq_objective(
    A: np.ndarray,
    P: np.ndarray,
    D: np.ndarray,
    Q: np.ndarray,
    alpha_P: float,
    alpha_D: float,
    alpha_Q: float,
) -> float:
    R = A - P @ D @ Q
    return float(
        0.5 * np.linalg.norm(R, ord="fro") ** 2
        + 0.5 * alpha_P * np.linalg.norm(P, ord="fro") ** 2
        + 0.5 * alpha_D * np.linalg.norm(D, ord="fro") ** 2
        + 0.5 * alpha_Q * np.linalg.norm(Q, ord="fro") ** 2
    )


def init_pdq_balanced(A: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    # vec(Gp D Gq + alpha_D D) = [Gq^T kron Gp + alpha_D I] vec(D)
    L = np.kron(Gq.T, Gp) + alpha_D * np.eye(r * r)
    d = solve(L, C.reshape(-1, order="F"), assume_a="pos")
    return d.reshape((r, r), order="F")


def clip_core_condition(D: np.ndarray, kappa_max: float) -> np.ndarray:
    U, s, Vt = np.linalg.svd(D, full_matrices=False)

    smax = max(float(s[0]), 1e-14)
    smin = smax / kappa_max
    s_clipped = np.maximum(s, smin)

    return U @ np.diag(s_clipped) @ Vt


def fit_pdq(
    A: np.ndarray,
    r: int,
    alpha_P: float,
    alpha_D: float,
    alpha_Q: float,
    max_sweeps: int,
    tol: float,
    conditioned_core: bool,
    kappa_core_max: float,
) -> tuple[np.ndarray, dict, pd.DataFrame]:
    P, D, Q = init_pdq_balanced(A, r)

    obj_prev = pdq_objective(A, P, D, Q, alpha_P, alpha_D, alpha_Q)
    history = []
    start = time.perf_counter()

    for sweep in range(1, max_sweeps + 1):
        P_old = P.copy()
        D_old = D.copy()
        Q_old = Q.copy()

        # P update
        HP = D @ Q @ Q.T @ D.T + alpha_P * np.eye(r)
        P = A @ Q.T @ D.T @ np.linalg.inv(HP)

        # D update
        D_candidate = solve_D_update(P, Q, A, alpha_D)

        if conditioned_core:
            D_projected = clip_core_condition(D_candidate, kappa_core_max)

            # Backtrack only if the projected core increases the objective at this substep.
            obj_before = pdq_objective(A, P, D_old, Q, alpha_P, alpha_D, alpha_Q)
            lam = 1.0
            for _ in range(25):
                D_try = (1.0 - lam) * D_old + lam * D_projected
                obj_try = pdq_objective(A, P, D_try, Q, alpha_P, alpha_D, alpha_Q)
                if obj_try <= obj_before or lam < 1e-8:
                    D = D_try
                    break
                lam *= 0.5
        else:
            D = D_candidate

        # Q update
        HQ = D.T @ P.T @ P @ D + alpha_Q * np.eye(r)
        RHS = D.T @ P.T @ A
        Q = np.linalg.solve(HQ, RHS)

        Ahat = P @ D @ Q
        obj = pdq_objective(A, P, D, Q, alpha_P, alpha_D, alpha_Q)

        step = (
            np.linalg.norm(P - P_old, ord="fro")
            + np.linalg.norm(D - D_old, ord="fro")
            + np.linalg.norm(Q - Q_old, ord="fro")
        ) / max(
            1.0,
            np.linalg.norm(P_old, ord="fro")
            + np.linalg.norm(D_old, ord="fro")
            + np.linalg.norm(Q_old, ord="fro"),
        )

        decrease = (obj_prev - obj) / max(1.0, abs(obj_prev))
        relerr = rel_fro_error(A, Ahat)
        kappa_D = safe_cond(D)

        history.append({
            "sweep": sweep,
            "objective": obj,
            "relative_error": relerr,
            "relative_step": step,
            "relative_objective_decrease": decrease,
            "kappa_D": kappa_D,
        })

        if max(abs(decrease), step) <= tol:
            break

        obj_prev = obj

    elapsed = time.perf_counter() - start
    hist = pd.DataFrame(history)

    info = {
        "time_sec": elapsed,
        "sweeps": len(history),
        "kappa_D": safe_cond(D),
        "final_step": float(hist["relative_step"].iloc[-1]) if len(hist) else np.nan,
        "final_obj_decrease": float(hist["relative_objective_decrease"].iloc[-1]) if len(hist) else np.nan,
    }

    return P @ D @ Q, info, hist


# ============================================================
# Evaluation
# ============================================================

def run_one_case(
    A0: np.ndarray,
    noise_level: float,
    rank: int,
    seed: int,
    config: dict,
    out_dir: Path,
) -> list[dict]:
    A = add_response_noise(A0, noise_level, seed)
    m, n = A.shape

    records = []

    methods = []

    Ahat, info = truncated_svd_approx(A, rank)
    methods.append(("Truncated SVD", Ahat, info))

    Ahat, info = randomized_svd_approx(
        A,
        r=rank,
        oversampling=config["randomized_oversampling"],
        n_iter=config["randomized_power_iter"],
        seed=seed,
    )
    methods.append(("Randomized SVD", Ahat, info))

    Ahat, info, hist = fit_pdq(
        A=A,
        r=rank,
        alpha_P=config["alpha_P"],
        alpha_D=config["alpha_D"],
        alpha_Q=config["alpha_Q"],
        max_sweeps=config["max_sweeps"],
        tol=config["tol"],
        conditioned_core=False,
        kappa_core_max=config["kappa_core_max"],
    )
    hist.to_csv(out_dir / "histories" / f"pdq_ridge_seed{seed}_noise{noise_level}_rank{rank}.csv", index=False)
    methods.append(("Proposed PDQ ridge", Ahat, info))

    Ahat, info, hist = fit_pdq(
        A=A,
        r=rank,
        alpha_P=config["alpha_P"],
        alpha_D=config["alpha_D"],
        alpha_Q=config["alpha_Q"],
        max_sweeps=config["max_sweeps"],
        tol=config["tol"],
        conditioned_core=True,
        kappa_core_max=config["kappa_core_max"],
    )
    hist.to_csv(out_dir / "histories" / f"pdq_conditioned_seed{seed}_noise{noise_level}_rank{rank}.csv", index=False)
    methods.append(("Proposed PDQ conditioned core", Ahat, info))

    # Ablation. Use this to support the role of side regularization,
    # not as a main competitor.
    Ahat, info, hist = fit_pdq(
        A=A,
        r=rank,
        alpha_P=config["weak_side_alpha"],
        alpha_D=config["alpha_D"],
        alpha_Q=config["weak_side_alpha"],
        max_sweeps=config["max_sweeps"],
        tol=config["tol"],
        conditioned_core=False,
        kappa_core_max=config["kappa_core_max"],
    )
    hist.to_csv(out_dir / "histories" / f"pdq_weak_side_seed{seed}_noise{noise_level}_rank{rank}.csv", index=False)
    methods.append(("PDQ weak-side-ridge ablation", Ahat, info))

    for method, Ahat, info in methods:
        records.append({
            "seed": seed,
            "noise_level": noise_level,
            "rank": rank,
            "method": method,
            "relerr_noisy": rel_fro_error(A, Ahat),
            "clean_error": rel_fro_error(A0, Ahat),
            "time_sec": info["time_sec"],
            "sweeps": info["sweeps"],
            "kappa_D": info["kappa_D"],
            "final_step": info["final_step"],
            "final_obj_decrease": info["final_obj_decrease"],
            "storage_ratio": storage_ratio_pdq(m, n, rank),
        })

    return records


def aggregate_results(all_runs: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["noise_level", "rank", "method"]

    rows = []
    for keys, g in all_runs.groupby(group_cols):
        noise, rank, method = keys
        rows.append({
            "noise_level": noise,
            "rank": rank,
            "method": method,
            "n_runs": len(g),
            "clean_error_mean": g["clean_error"].mean(),
            "clean_error_ci95": confidence_half_width(g["clean_error"]),
            "relerr_noisy_mean": g["relerr_noisy"].mean(),
            "relerr_noisy_ci95": confidence_half_width(g["relerr_noisy"]),
            "time_sec_mean": g["time_sec"].mean(),
            "time_sec_ci95": confidence_half_width(g["time_sec"]),
            "kappa_D_mean": g["kappa_D"].replace([np.inf, -np.inf], np.nan).mean(),
            "kappa_D_max": g["kappa_D"].replace([np.inf, -np.inf], np.nan).max(),
            "sweeps_mean": g["sweeps"].mean(),
            "storage_ratio_mean": g["storage_ratio"].mean(),
        })

    return pd.DataFrame(rows).sort_values(["noise_level", "rank", "method"])


def paired_comparisons(all_runs: pd.DataFrame) -> pd.DataFrame:
    """
    Compare each method against Truncated SVD on the same seed/noise/rank case.
    """
    key_cols = ["seed", "noise_level", "rank"]
    base = all_runs[all_runs["method"] == "Truncated SVD"][
        key_cols + ["clean_error", "relerr_noisy", "time_sec"]
    ].rename(columns={
        "clean_error": "svd_clean_error",
        "relerr_noisy": "svd_relerr_noisy",
        "time_sec": "svd_time_sec",
    })

    merged = all_runs.merge(base, on=key_cols, how="left")
    merged["clean_error_gain_vs_svd_pct"] = (
        100.0 * (merged["svd_clean_error"] - merged["clean_error"]) / merged["svd_clean_error"]
    )
    merged["noisy_error_gain_vs_svd_pct"] = (
        100.0 * (merged["svd_relerr_noisy"] - merged["relerr_noisy"]) / merged["svd_relerr_noisy"]
    )
    merged["runtime_multiple_vs_svd"] = merged["time_sec"] / merged["svd_time_sec"].replace(0, np.nan)

    rows = []
    for method, g in merged.groupby("method"):
        if method == "Truncated SVD":
            continue
        rows.append({
            "method": method,
            "n_paired_cases": len(g),
            "mean_clean_gain_vs_svd_pct": g["clean_error_gain_vs_svd_pct"].mean(),
            "median_clean_gain_vs_svd_pct": g["clean_error_gain_vs_svd_pct"].median(),
            "clean_win_rate_vs_svd": float((g["clean_error_gain_vs_svd_pct"] > 0).mean()),
            "mean_noisy_gain_vs_svd_pct": g["noisy_error_gain_vs_svd_pct"].mean(),
            "median_runtime_multiple_vs_svd": g["runtime_multiple_vs_svd"].median(),
            "mean_kappa_D": g["kappa_D"].replace([np.inf, -np.inf], np.nan).mean(),
            "max_kappa_D": g["kappa_D"].replace([np.inf, -np.inf], np.nan).max(),
        })

    return pd.DataFrame(rows).sort_values("method")


def make_manuscript_table(aggregate: pd.DataFrame, out_dir: Path) -> None:
    """
    A compact table for the paper.

    We keep only the main method, SVD baselines, and key evidence.
    """
    keep_methods = [
        "Truncated SVD",
        "Randomized SVD",
        "Proposed PDQ ridge",
        "Proposed PDQ conditioned core",
    ]

    tab = aggregate[aggregate["method"].isin(keep_methods)].copy()
    tab = tab[
        [
            "noise_level", "rank", "method", "clean_error_mean", "clean_error_ci95",
            "relerr_noisy_mean", "time_sec_mean", "kappa_D_mean", "storage_ratio_mean"
        ]
    ]

    tex = tab.to_latex(
        index=False,
        float_format=lambda x: f"{x:.4e}" if np.isfinite(x) else r"$\infty$",
        caption=(
            "Application-derived structural-response validation over multiple seeds, "
            "ranks, and noise levels. The response snapshots are generated by solving "
            "multiple load cases using a real SuiteSparse structural stiffness matrix."
        ),
        label="tab:industrial_structural_response_multirun",
    )

    (out_dir / "manuscript_table.tex").write_text(tex, encoding="utf-8")


def make_figures(aggregate: pd.DataFrame, paired: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"

    # 1. Clean error by method at each noise level, averaged across ranks.
    g = aggregate.groupby(["noise_level", "method"], as_index=False)["clean_error_mean"].mean()
    pivot = g.pivot(index="noise_level", columns="method", values="clean_error_mean")

    plt.figure(figsize=(9, 5))
    for col in pivot.columns:
        if "ablation" in col:
            continue
        plt.plot(pivot.index, pivot[col], marker="o", label=col)
    plt.xlabel("Noise level")
    plt.ylabel("Mean clean-response error")
    plt.title("Denoising performance on real structural-response data")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "summary_clean_error.png", dpi=300)
    plt.close()

    # 2. Runtime by method, log scale.
    g = aggregate.groupby("method", as_index=False)["time_sec_mean"].median()
    plt.figure(figsize=(8, 4.5))
    plt.bar(g["method"], g["time_sec_mean"])
    plt.yscale("log")
    plt.ylabel("Median runtime over cases, seconds, log scale")
    plt.xticks(rotation=25, ha="right")
    plt.title("Runtime trade-off")
    plt.tight_layout()
    plt.savefig(fig_dir / "summary_runtime.png", dpi=300)
    plt.close()

    # 3. Core condition number for PDQ variants.
    k = aggregate[aggregate["method"].str.contains("PDQ", regex=False)].copy()
    k = k.groupby("method", as_index=False)["kappa_D_mean"].median()
    plt.figure(figsize=(8, 4.5))
    plt.bar(k["method"], k["kappa_D_mean"])
    plt.yscale("log")
    plt.ylabel(r"Median learned core condition number $\kappa(D)$")
    plt.xticks(rotation=25, ha="right")
    plt.title("Core-conditioning diagnostic")
    plt.tight_layout()
    plt.savefig(fig_dir / "summary_kappa.png", dpi=300)
    plt.close()

    # 4. Paired clean improvement table as plot.
    if not paired.empty:
        plt.figure(figsize=(8, 4.5))
        plt.bar(paired["method"], paired["mean_clean_gain_vs_svd_pct"])
        plt.axhline(0, linewidth=1)
        plt.ylabel("Mean clean-error gain vs truncated SVD (%)")
        plt.xticks(rotation=25, ha="right")
        plt.title("Paired denoising gain over SVD")
        plt.tight_layout()
        plt.savefig(fig_dir / "summary_paired_gain.png", dpi=300)
        plt.close()


def run_experiment(config: dict) -> None:
    out_dir = make_output_dir(config["out_root"])
    save_config(config, out_dir)

    print("=" * 72)
    print("INDUSTRIAL PDQ VALIDATION V2")
    print("=" * 72)
    print(f"[INFO] Output folder: {out_dir.resolve()}")

    K = load_suitesparse_matrix(config, out_dir)

    dataset_rows = [{
        "matrix_group": config["matrix"]["group"],
        "matrix_name": config["matrix"]["name"],
        "matrix_description": config["matrix"]["description"],
        "K_rows": K.shape[0],
        "K_cols": K.shape[1],
        "K_nnz": K.nnz,
        "n_scenarios": config["n_scenarios"],
        "n_base_loads": config["n_base_loads"],
        "load_sparsity": config["load_sparsity"],
        "ranks": str(config["ranks"]),
        "noise_levels": str(config["noise_levels"]),
        "seeds": str(config["seeds"]),
    }]
    pd.DataFrame(dataset_rows).to_csv(out_dir / "dataset_summary.csv", index=False)

    all_records = []

    # Build one clean response matrix per seed. This avoids pretending that
    # repeated noise on the same response field is independent industrial evidence.
    for seed in config["seeds"]:
        print("\n" + "-" * 72)
        print(f"[INFO] Building clean response matrix for seed={seed}")
        print("-" * 72)

        A0 = build_response_matrix(K, config, seed=seed)

        for noise_level in config["noise_levels"]:
            for rank in config["ranks"]:
                print(f"[INFO] Case: seed={seed}, noise={noise_level}, rank={rank}")
                case_records = run_one_case(
                    A0=A0,
                    noise_level=noise_level,
                    rank=rank,
                    seed=seed,
                    config=config,
                    out_dir=out_dir,
                )
                all_records.extend(case_records)

    all_runs = pd.DataFrame(all_records)
    all_runs.to_csv(out_dir / "all_runs.csv", index=False)

    aggregate = aggregate_results(all_runs)
    aggregate.to_csv(out_dir / "aggregate_table.csv", index=False)

    paired = paired_comparisons(all_runs)
    paired.to_csv(out_dir / "paired_comparison_table.csv", index=False)

    make_manuscript_table(aggregate, out_dir)
    make_figures(aggregate, paired, out_dir)

    print("\n" + "=" * 72)
    print("KEY PAIRED COMPARISON")
    print("=" * 72)
    print(paired.to_string(index=False))

    print("\n" + "=" * 72)
    print("SAVED OUTPUTS")
    print("=" * 72)
    for name in [
        "dataset_summary.csv",
        "all_runs.csv",
        "aggregate_table.csv",
        "paired_comparison_table.csv",
        "manuscript_table.tex",
        "figures/summary_clean_error.png",
        "figures/summary_runtime.png",
        "figures/summary_kappa.png",
        "figures/summary_paired_gain.png",
        "config.json",
    ]:
        print(out_dir / name)

    print("\n[INFO] Done.")


if __name__ == "__main__":
    run_experiment(CONFIG)