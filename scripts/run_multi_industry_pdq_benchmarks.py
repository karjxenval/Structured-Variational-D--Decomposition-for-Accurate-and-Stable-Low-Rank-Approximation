#!/usr/bin/env python3
"""Run a multi-industry PDQ surrogate benchmark with baselines and ablations.

This is the new reviewer-facing validation script for the repository.

Industrial cases
----------------
1. Structural truss bridge segment: finite-element stiffness model with moving
   deck loads, wind loads, and sparse displacement sensors.
2. Thermal process line: steady heat-conduction operator with localized heaters.
3. DC power-grid load flow: reduced bus-angle operator with time-varying loads
   and renewable injections.

Methods
-------
- Mean response baseline
- Truncated SVD / POD
- Randomized SVD
- PDQ ridge
- PDQ conditioned core
- PDQ weak-side-ridge ablation
- PDQ random-init ablation

Sensor policies
---------------
- Variance sensors: high-variance rows chosen from training responses.
- Random sensors: ablation showing why informed sensor placement matters.

Outputs
-------
results_multi_industry_pdq_<timestamp>/
    config.json
    all_runs.csv
    aggregate_results.csv
    main_table.csv
    main_table.tex
    histories/*.csv
    figures/*.png
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pdqbench.cases import add_measurement_noise, build_all_cases
from pdqbench.metrics import physical_response_metrics, rel_fro_error
from pdqbench.pdq import fit_all_methods
from pdqbench.reporting import aggregate_results, make_main_table, save_plots
from pdqbench.sensors import reconstruct_from_sensors, select_random_sensors, select_variance_sensors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", default="results_multi_industry_pdq", help="Root name for timestamped output folder.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 29, 47], help="Random seeds.")
    parser.add_argument("--ranks", type=int, nargs="+", default=[4, 8, 12], help="Low-rank dimensions.")
    parser.add_argument("--sensor-counts", type=int, nargs="+", default=[6, 10, 16], help="Sparse sensor counts.")
    parser.add_argument("--training-noise", type=float, nargs="+", default=[0.0, 0.01, 0.03], help="Training snapshot noise levels.")
    parser.add_argument("--sensor-noise", type=float, nargs="+", default=[0.0, 0.01, 0.03], help="Test sensor noise levels.")
    parser.add_argument("--n-train", type=int, default=80, help="Training scenarios per case.")
    parser.add_argument("--n-test", type=int, default=40, help="Test scenarios per case.")
    parser.add_argument("--coefficient-ridge", type=float, default=1e-8, help="Ridge used for online sensor coefficient recovery.")
    parser.add_argument("--alpha-p", type=float, default=1e-4)
    parser.add_argument("--alpha-d", type=float, default=1e-4)
    parser.add_argument("--alpha-q", type=float, default=1e-4)
    parser.add_argument("--weak-side-alpha", type=float, default=1e-8)
    parser.add_argument("--kappa-core-max", type=float, default=50.0)
    parser.add_argument("--max-sweeps", type=int, default=120)
    parser.add_argument("--tol", type=float, default=1e-7)
    parser.add_argument("--oversampling", type=int, default=10)
    parser.add_argument("--power-iter", type=int, default=2)
    parser.add_argument("--smoke", action="store_true", help="Small run for CI/local testing.")
    return parser.parse_args()


def make_output_dir(out_root: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"{out_root}_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "histories").mkdir(exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    return out_dir


def save_config(args: argparse.Namespace, out_dir: Path, pdq_cfg: dict) -> None:
    payload = vars(args).copy()
    payload["pdq_config"] = pdq_cfg
    payload["system"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (out_dir / "config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def evaluate_sensor_recovery(case, fitted, U_test_sensor_noisy: np.ndarray, sensor_rows: np.ndarray, ridge: float) -> dict:
    metric_rows = []
    start = time.perf_counter()
    for j in range(case.test_states.shape[1]):
        y = U_test_sensor_noisy[sensor_rows, j]
        u_hat = reconstruct_from_sensors(fitted.mean, fitted.basis, sensor_rows, y, ridge)
        metric_rows.append(physical_response_metrics(case.operator, case.test_states[:, j], u_hat, case.test_forces[:, j]))
    online_time_per_case = (time.perf_counter() - start) / max(1, case.test_states.shape[1])
    metrics = pd.DataFrame(metric_rows)
    return {
        "sensor_rel_state_error": float(metrics["rel_state_error"].mean()),
        "sensor_rel_max_state_error": float(metrics["rel_max_state_error"].mean()),
        "sensor_rel_energy_error": float(metrics["rel_energy_error"].mean()),
        "sensor_rel_energy_norm_error": float(metrics["rel_energy_norm_error"].mean()),
        "sensor_rel_operator_residual": float(metrics["rel_operator_residual"].mean()),
        "online_time_per_case_sec": online_time_per_case,
        "online_speedup_vs_direct_solve": case.direct_solve_time_per_case_sec / max(online_time_per_case, 1e-30),
    }


def run() -> Path:
    args = parse_args()
    if args.smoke:
        args.seeds = [13]
        args.ranks = [3]
        args.sensor_counts = [4]
        args.training_noise = [0.0]
        args.sensor_noise = [0.0]
        args.n_train = 18
        args.n_test = 8
        args.max_sweeps = 20

    pdq_cfg = {
        "alpha_P": args.alpha_p,
        "alpha_D": args.alpha_d,
        "alpha_Q": args.alpha_q,
        "weak_side_alpha": args.weak_side_alpha,
        "kappa_core_max": args.kappa_core_max,
        "max_sweeps": args.max_sweeps,
        "tol": args.tol,
        "oversampling": args.oversampling,
        "power_iter": args.power_iter,
    }

    out_dir = make_output_dir(args.out_root)
    save_config(args, out_dir, pdq_cfg)

    records: list[dict] = []
    case_metadata = []

    for seed in args.seeds:
        cases = build_all_cases(args.n_train, args.n_test, seed)
        for case in cases:
            print(f"\n[CASE] {case.name} | seed={seed} | state_dim={case.train_states.shape[0]}")
            case_metadata.append({"seed": seed, "case": case.name, **case.metadata})

            for training_noise in args.training_noise:
                U_train_noisy = add_measurement_noise(case.train_states, training_noise, seed)
                for rank in args.ranks:
                    print(f"  fitting rank={rank}, train_noise={training_noise}")
                    fitted_models, histories = fit_all_methods(U_train_noisy, rank, seed, pdq_cfg)
                    for method, hist in histories.items():
                        safe_name = method.lower().replace("/", "_").replace(" ", "_")
                        hist.to_csv(
                            out_dir / "histories" / f"{case.name}_seed{seed}_noise{training_noise}_rank{rank}_{safe_name}.csv",
                            index=False,
                        )

                    for sensor_noise in args.sensor_noise:
                        U_test_sensor_noisy = add_measurement_noise(case.test_states, sensor_noise, seed + 100)
                        for sensor_count in args.sensor_counts:
                            sensor_policies = {
                                "variance": select_variance_sensors(case.train_states, sensor_count),
                                "random": select_random_sensors(case.train_states.shape[0], sensor_count, seed),
                            }
                            for sensor_policy, sensor_rows in sensor_policies.items():
                                for fitted in fitted_models:
                                    full_test = fitted.reconstruct_full(case.test_states)
                                    sensor_metrics = evaluate_sensor_recovery(
                                        case,
                                        fitted,
                                        U_test_sensor_noisy,
                                        sensor_rows,
                                        args.coefficient_ridge,
                                    )
                                    records.append({
                                        "case": case.name,
                                        "seed": seed,
                                        "rank": rank,
                                        "training_noise": training_noise,
                                        "sensor_noise": sensor_noise,
                                        "sensor_count": sensor_count,
                                        "sensor_fraction": sensor_count / case.train_states.shape[0],
                                        "sensor_policy": sensor_policy,
                                        "method": fitted.method,
                                        "train_clean_rel_error": rel_fro_error(case.train_states, fitted.Ahat_train),
                                        "train_noisy_rel_error": rel_fro_error(U_train_noisy, fitted.Ahat_train),
                                        "full_test_rel_error": rel_fro_error(case.test_states, full_test),
                                        "offline_fit_time_sec": fitted.time_sec,
                                        "storage_ratio": fitted.storage_ratio,
                                        "kappa_D": fitted.kappa_D,
                                        "sweeps": fitted.sweeps,
                                        "final_step": fitted.final_step,
                                        "final_obj_decrease": fitted.final_obj_decrease,
                                        "direct_solve_time_per_case_sec": case.direct_solve_time_per_case_sec,
                                        **sensor_metrics,
                                    })

    all_runs = pd.DataFrame(records)
    metadata = pd.DataFrame(case_metadata).drop_duplicates()
    agg = aggregate_results(all_runs)
    main_table = make_main_table(agg)

    all_runs.to_csv(out_dir / "all_runs.csv", index=False)
    metadata.to_csv(out_dir / "case_metadata.csv", index=False)
    agg.to_csv(out_dir / "aggregate_results.csv", index=False)
    main_table.to_csv(out_dir / "main_table.csv", index=False)
    main_table.to_latex(out_dir / "main_table.tex", index=False, float_format="%.4e")
    save_plots(agg, out_dir)

    print(f"\n[DONE] Results written to: {out_dir}")
    return out_dir


if __name__ == "__main__":
    run()
