"""
Real industrial/scientific-computing validation for core-conditioned PDQ factorization.

What this script does
---------------------
1. Downloads a real structural-engineering stiffness matrix from SuiteSparse:
   HB/bcsstk18, R.E. Ginna Nuclear Power Station stiffness matrix.

2. Builds a simulation-response matrix:
       K x_j = b_j,    j = 1,...,n_scenarios
       A = [x_1, ..., x_n]

   Here K is the real sparse stiffness matrix, b_j are load cases,
   and x_j are displacement/response fields.

3. Addss controlled sensor/simulation noise.

4. Compares:
   - Truncated SVD
   - Randomized SVD
   - Proposed PDQ ridge
   - Proposed PDQ with bounded core

5. Reports only the essential evidence:
   - reconstruction error on noisy data
   - clean-response error
   - core condition number kappa(D)
   - runtime
   - sweeps
   - storage ratio
   - convergence diagnostics

Outputs
-------
results_industrial_pdq/
    dataset_summary.csv
    performance_table.csv
    convergence_pdq_ridge.csv
    convergence_pdq_conditioned.csv
    real_case_summary.png
    latex_performance_table.txt
"""

import os
import time
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import mmread
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.linalg import solve


# ============================================================
# Configuration
# ============================================================

CONFIG = {
    "out_dir": "results_industrial_pdq",

    # Real structural engineering matrix from SuiteSparse:
    # https://sparse.tamu.edu/HB/bcsstk18
    "matrix_group": "HB",
    "matrix_name": "bcsstk18",
    "matrix_url": "https://sparse.tamu.edu/MM/HB/bcsstk18.tar.gz",

    # Response-matrix construction
    "n_scenarios": 80,
    "n_base_loads": 12,
    "load_sparsity": 35,
    "seed": 13,

    # Low-rank approximation
    "rank": 20,

    # Noise level in response snapshots
    # This is relative to entrywise RMS of the clean response matrix.
    "noise_level": 0.02,

    # PDQ parameters
    "alpha_P": 1e-4,
    "alpha_D": 1e-4,
    "alpha_Q": 1e-4,
    "max_sweeps": 150,
    "tol": 1e-7,

    # Core conditioning
    # The conditioned variant enforces kappa(D) <= kappa_core_max.
    "kappa_core_max": 1e3,

    # Small diagonal shift for numerical safety in sparse solve.
    # This is not part of the factorization. It only stabilizes
    # the construction of simulation responses from the public matrix.
    "solve_shift_scale": 1e-10,
}


# ============================================================
# Utility functions
# ============================================================

def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[INFO] Found existing download: {dest}")
        return

    print(f"[INFO] Downloading:\n  {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"[INFO] Saved to: {dest}")


def extract_matrix_market(tar_path: Path, extract_dir: Path) -> Path:
    marker = extract_dir / ".extracted"
    if marker.exists():
        mtx_files = list(extract_dir.rglob("*.mtx"))
        if mtx_files:
            return mtx_files[0]

    print(f"[INFO] Extracting: {tar_path}")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(extract_dir)

    marker.write_text("done")

    mtx_files = list(extract_dir.rglob("*.mtx"))
    if not mtx_files:
        raise FileNotFoundError("No .mtx file found after extraction.")

    print(f"[INFO] Matrix Market file: {mtx_files[0]}")
    return mtx_files[0]


def load_suitesparse_matrix(config: dict, out_dir: Path) -> sparse.csc_matrix:
    data_dir = ensure_dir(out_dir / "data")
    tar_path = data_dir / f"{config['matrix_name']}.tar.gz"
    extract_dir = data_dir / config["matrix_name"]

    download_file(config["matrix_url"], tar_path)
    mtx_path = extract_matrix_market(tar_path, extract_dir)

    print("[INFO] Reading Matrix Market file...")
    K = mmread(str(mtx_path)).tocsc().astype(float)

    # Ensure exact symmetry numerically. SuiteSparse metadata says bcsstk18 is symmetric.
    K = 0.5 * (K + K.T)
    K = K.tocsc()

    print(f"[INFO] Loaded K with shape={K.shape}, nnz={K.nnz}")
    return K


def add_small_diagonal_shift(K: sparse.csc_matrix, shift_scale: float) -> sparse.csc_matrix:
    diag_abs_mean = np.mean(np.abs(K.diagonal()))
    if diag_abs_mean <= 0 or not np.isfinite(diag_abs_mean):
        diag_abs_mean = 1.0

    shift = shift_scale * diag_abs_mean
    print(f"[INFO] Adding diagonal shift for solve stability: {shift:.3e}")
    return (K + shift * sparse.eye(K.shape[0], format="csc")).tocsc()


def make_structured_loads(
    n_dof: int,
    n_scenarios: int,
    n_base_loads: int,
    load_sparsity: int,
    seed: int,
) -> np.ndarray:
    """
    Build load vectors that mimic industrial load cases.

    Each scenario is a combination of a small number of localized load patterns.
    This creates a response matrix with approximate low-dimensional structure,
    which is typical in reduced-order modelling and digital-twin workflows.
    """
    rng = np.random.default_rng(seed)

    B0 = np.zeros((n_dof, n_base_loads))
    for k in range(n_base_loads):
        idx = rng.choice(n_dof, size=load_sparsity, replace=False)
        values = rng.normal(loc=0.0, scale=1.0, size=load_sparsity)
        B0[idx, k] = values

    # Operating-scenario coefficients.
    C = rng.normal(size=(n_base_loads, n_scenarios))

    # Add mild smooth trend across scenarios to mimic changing operating regimes.
    t = np.linspace(0.0, 1.0, n_scenarios)
    for k in range(n_base_loads):
        C[k, :] += 0.5 * np.sin(2 * np.pi * (k + 1) * t / max(2, n_base_loads))

    B = B0 @ C

    # Normalize load scale.
    norm_B = np.linalg.norm(B, ord="fro")
    if norm_B > 0:
        B = B / norm_B * np.sqrt(n_scenarios)

    return B


def build_response_matrix(K: sparse.csc_matrix, config: dict) -> np.ndarray:
    """
    Solve K X = B for many load cases and return A = X.
    """
    n = K.shape[0]

    B = make_structured_loads(
        n_dof=n,
        n_scenarios=config["n_scenarios"],
        n_base_loads=config["n_base_loads"],
        load_sparsity=config["load_sparsity"],
        seed=config["seed"],
    )

    K_solve = add_small_diagonal_shift(K, config["solve_shift_scale"])

    print("[INFO] Factoring sparse stiffness matrix...")
    start = time.perf_counter()
    lu = splu(K_solve)
    factor_time = time.perf_counter() - start
    print(f"[INFO] Sparse LU factorization time: {factor_time:.2f} seconds")

    print("[INFO] Solving for response snapshots...")
    start = time.perf_counter()
    X = lu.solve(B)
    solve_time = time.perf_counter() - start
    print(f"[INFO] Multiple-RHS solve time: {solve_time:.2f} seconds")

    # Remove global mean and normalize for numerical comparability.
    A = X - X.mean(axis=1, keepdims=True)
    fro = np.linalg.norm(A, ord="fro")
    if fro > 0:
        A = A / fro

    return A


def add_noise(A_clean: np.ndarray, noise_level: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 999)
    rms = np.linalg.norm(A_clean, ord="fro") / np.sqrt(A_clean.size)
    noise = noise_level * rms * rng.normal(size=A_clean.shape)
    return A_clean + noise


def storage_ratio_pdq(m: int, n: int, r: int) -> float:
    return (m * r + r * r + r * n) / (m * n)


def rel_fro_error(A: np.ndarray, Ahat: np.ndarray) -> float:
    denom = np.linalg.norm(A, ord="fro")
    if denom == 0:
        return np.nan
    return np.linalg.norm(A - Ahat, ord="fro") / denom


def safe_cond(D: np.ndarray) -> float:
    s = np.linalg.svd(D, compute_uv=False)
    if s[-1] <= 1e-15:
        return np.inf
    return float(s[0] / s[-1])


# ============================================================
# Baselines
# ============================================================

def truncated_svd_approx(A: np.ndarray, r: int) -> tuple[np.ndarray, dict]:
    start = time.perf_counter()
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    Ahat = (U[:, :r] * s[:r]) @ Vt[:r, :]
    elapsed = time.perf_counter() - start

    info = {
        "time_sec": elapsed,
        "sweeps": 0,
        "kappa_D": np.nan,
    }
    return Ahat, info


def randomized_svd_approx(
    A: np.ndarray,
    r: int,
    oversampling: int = 10,
    n_iter: int = 2,
    seed: int = 13,
) -> tuple[np.ndarray, dict]:
    """
    Simple randomized SVD implementation.

    This is included to avoid relying on scikit-learn.
    """
    rng = np.random.default_rng(seed)
    m, n = A.shape
    ell = min(n, r + oversampling)

    start = time.perf_counter()

    Omega = rng.normal(size=(n, ell))
    Y = A @ Omega

    for _ in range(n_iter):
        Y = A @ (A.T @ Y)

    Q, _ = np.linalg.qr(Y, mode="reduced")
    B = Q.T @ A

    Ub, s, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ Ub

    Ahat = (U[:, :r] * s[:r]) @ Vt[:r, :]
    elapsed = time.perf_counter() - start

    info = {
        "time_sec": elapsed,
        "sweeps": 0,
        "kappa_D": np.nan,
    }
    return Ahat, info


# ============================================================
# PDQ method
# ============================================================

def pdq_objective(A: np.ndarray, P: np.ndarray, D: np.ndarray, Q: np.ndarray,
                  alpha_P: float, alpha_D: float, alpha_Q: float) -> float:
    R = A - P @ D @ Q
    return (
        0.5 * np.linalg.norm(R, ord="fro") ** 2
        + 0.5 * alpha_P * np.linalg.norm(P, ord="fro") ** 2
        + 0.5 * alpha_D * np.linalg.norm(D, ord="fro") ** 2
        + 0.5 * alpha_Q * np.linalg.norm(Q, ord="fro") ** 2
    )


def init_pdq_from_svd(A: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    U = U[:, :r]
    s = s[:r]
    Vt = Vt[:r, :]

    # Balanced initialization:
    # A_r = (U sqrt(S)) I (sqrt(S) V^T)
    sqrt_s = np.sqrt(np.maximum(s, 1e-15))
    P = U * sqrt_s
    D = np.eye(r)
    Q = sqrt_s[:, None] * Vt

    return P, D, Q


def solve_D_update(P: np.ndarray, Q: np.ndarray, A: np.ndarray, alpha_D: float) -> np.ndarray:
    """
    Solve:
        (P^T P) D (Q Q^T) + alpha_D D = P^T A Q^T

    Vectorized as:
        [ (Q Q^T)^T kron (P^T P) + alpha_D I ] vec(D) = vec(P^T A Q^T)
    """
    r = P.shape[1]
    Gp = P.T @ P
    Gq = Q @ Q.T
    C = P.T @ A @ Q.T

    L = np.kron(Gq.T, Gp) + alpha_D * np.eye(r * r)
    d = solve(L, C.reshape(-1, order="F"), assume_a="pos")
    D = d.reshape((r, r), order="F")
    return D


def clip_core_condition(D: np.ndarray, kappa_max: float) -> np.ndarray:
    """
    Enforce kappa(D) <= kappa_max by clipping singular values.

    The upper bound is taken from the current largest singular value.
    The lower bound is upper / kappa_max.
    """
    U, s, Vt = np.linalg.svd(D, full_matrices=False)

    smax = max(float(s[0]), 1e-12)
    smin_allowed = smax / kappa_max
    s_clipped = np.maximum(s, smin_allowed)

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
    m, n = A.shape

    P, D, Q = init_pdq_from_svd(A, r)
    obj_prev = pdq_objective(A, P, D, Q, alpha_P, alpha_D, alpha_Q)

    history = []
    start = time.perf_counter()

    for sweep in range(1, max_sweeps + 1):
        P_old, D_old, Q_old = P.copy(), D.copy(), Q.copy()

        # P update
        H_P = D @ Q @ Q.T @ D.T + alpha_P * np.eye(r)
        P = A @ Q.T @ D.T @ np.linalg.inv(H_P)

        # D update
        D_candidate = solve_D_update(P, Q, A, alpha_D)

        if conditioned_core:
            D_projected = clip_core_condition(D_candidate, kappa_core_max)

            # Simple backtracking to avoid objective increase at the D step.
            obj_before_D = pdq_objective(A, P, D_old, Q, alpha_P, alpha_D, alpha_Q)
            lam = 1.0
            accepted = False
            for _ in range(20):
                D_try = (1.0 - lam) * D_old + lam * D_projected
                obj_try = pdq_objective(A, P, D_try, Q, alpha_P, alpha_D, alpha_Q)
                if obj_try <= obj_before_D or lam < 1e-6:
                    D = D_try
                    accepted = True
                    break
                lam *= 0.5

            if not accepted:
                D = D_old
        else:
            D = D_candidate

        # Q update
        H_Q = D.T @ P.T @ P @ D + alpha_Q * np.eye(r)
        RHS_Q = D.T @ P.T @ A
        Q = np.linalg.solve(H_Q, RHS_Q)

        obj = pdq_objective(A, P, D, Q, alpha_P, alpha_D, alpha_Q)
        Ahat = P @ D @ Q
        relerr = rel_fro_error(A, Ahat)

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
    Ahat = P @ D @ Q
    hist_df = pd.DataFrame(history)

    info = {
        "time_sec": elapsed,
        "sweeps": len(history),
        "kappa_D": safe_cond(D),
        "P": P,
        "D": D,
        "Q": Q,
    }

    return Ahat, info, hist_df


# ============================================================
# Main experiment
# ============================================================

def run_experiment(config: dict) -> None:
    out_dir = ensure_dir(config["out_dir"])

    print("\n==================================================")
    print(" REAL INDUSTRIAL PDQ VALIDATION")
    print("==================================================\n")

    K = load_suitesparse_matrix(config, out_dir)

    A_clean = build_response_matrix(K, config)
    A_noisy = add_noise(A_clean, config["noise_level"], config["seed"])

    m, n = A_noisy.shape
    r = config["rank"]

    print(f"[INFO] Response matrix A shape: {A_noisy.shape}")
    print(f"[INFO] Target rank r: {r}")
    print(f"[INFO] Storage ratio PDQ: {storage_ratio_pdq(m, n, r):.4f}")

    dataset_summary = pd.DataFrame([{
        "matrix_group": config["matrix_group"],
        "matrix_name": config["matrix_name"],
        "K_rows": K.shape[0],
        "K_cols": K.shape[1],
        "K_nnz": K.nnz,
        "response_rows": m,
        "response_cols": n,
        "rank_used": r,
        "noise_level": config["noise_level"],
        "storage_ratio_PDQ": storage_ratio_pdq(m, n, r),
    }])
    dataset_summary.to_csv(out_dir / "dataset_summary.csv", index=False)

    methods = []

    print("\n[INFO] Running truncated SVD...")
    Ahat, info = truncated_svd_approx(A_noisy, r)
    methods.append(("Truncated SVD", Ahat, info))

    print("[INFO] Running randomized SVD...")
    Ahat, info = randomized_svd_approx(A_noisy, r, seed=config["seed"])
    methods.append(("Randomized SVD", Ahat, info))

    print("[INFO] Running proposed PDQ ridge...")
    Ahat, info, hist_ridge = fit_pdq(
        A=A_noisy,
        r=r,
        alpha_P=config["alpha_P"],
        alpha_D=config["alpha_D"],
        alpha_Q=config["alpha_Q"],
        max_sweeps=config["max_sweeps"],
        tol=config["tol"],
        conditioned_core=False,
        kappa_core_max=config["kappa_core_max"],
    )
    methods.append(("Proposed PDQ ridge", Ahat, info))
    hist_ridge.to_csv(out_dir / "convergence_pdq_ridge.csv", index=False)

    print("[INFO] Running proposed PDQ conditioned core...")
    Ahat, info, hist_cond = fit_pdq(
        A=A_noisy,
        r=r,
        alpha_P=config["alpha_P"],
        alpha_D=config["alpha_D"],
        alpha_Q=config["alpha_Q"],
        max_sweeps=config["max_sweeps"],
        tol=config["tol"],
        conditioned_core=True,
        kappa_core_max=config["kappa_core_max"],
    )
    methods.append(("Proposed PDQ conditioned core", Ahat, info))
    hist_cond.to_csv(out_dir / "convergence_pdq_conditioned.csv", index=False)

    rows = []
    for name, Ahat, info in methods:
        rows.append({
            "Method": name,
            "RelErr_to_noisy_A": rel_fro_error(A_noisy, Ahat),
            "CleanErr_to_response_A0": rel_fro_error(A_clean, Ahat),
            "kappa_D": info["kappa_D"],
            "Time_sec": info["time_sec"],
            "Sweeps": info["sweeps"],
            "Storage_ratio": storage_ratio_pdq(m, n, r),
        })

    perf = pd.DataFrame(rows)
    perf.to_csv(out_dir / "performance_table.csv", index=False)

    # Save LaTeX table for direct manuscript use.
    latex_table = perf.to_latex(
        index=False,
        float_format=lambda x: f"{x:.4e}" if np.isfinite(x) else r"$\infty$",
        caption=(
            "Application-derived structural response validation. "
            "The response matrix is generated by solving multiple load cases "
            "using a real SuiteSparse structural stiffness matrix."
        ),
        label="tab:real_structural_response_validation",
    )
    (out_dir / "latex_performance_table.txt").write_text(latex_table)

    print("\n==================================================")
    print(" DATASET SUMMARY")
    print("==================================================")
    print(dataset_summary.to_string(index=False))

    print("\n==================================================")
    print(" PERFORMANCE TABLE")
    print("==================================================")
    print(perf.to_string(index=False))

    # Essential plot only.
    plot_summary(perf, out_dir)

    print("\n==================================================")
    print(" DONE")
    print("==================================================")
    print(f"Results saved in: {out_dir.resolve()}")
    print("\nKey files:")
    print(f"  {out_dir / 'dataset_summary.csv'}")
    print(f"  {out_dir / 'performance_table.csv'}")
    print(f"  {out_dir / 'real_case_summary.png'}")
    print(f"  {out_dir / 'latex_performance_table.txt'}")


def plot_summary(perf: pd.DataFrame, out_dir: Path) -> None:
    """
    One compact plot:
    left axis: clean-response error
    right axis: core condition number for PDQ methods.
    """
    fig, ax1 = plt.subplots(figsize=(10, 5))

    x = np.arange(len(perf))
    labels = perf["Method"].tolist()

    ax1.bar(x, perf["CleanErr_to_response_A0"].values)
    ax1.set_ylabel("Clean-response relative error")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right")
    ax1.set_title("Real structural-response validation")

    ax2 = ax1.twinx()
    kappa = perf["kappa_D"].replace([np.inf, -np.inf], np.nan).values
    ax2.plot(x, kappa, marker="o", linewidth=2)
    ax2.set_ylabel(r"Core condition number $\kappa(D)$")

    fig.tight_layout()
    fig.savefig(out_dir / "real_case_summary.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    run_experiment(CONFIG)