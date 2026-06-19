"""Industry-facing benchmark case generators.

The cases are deterministic, physically structured digital-twin surrogates:
structural FE, thermal process, and DC load-flow grids. They are not toy i.i.d.
Gaussian matrices; each produces an operator K, forcing scenarios F, and state
snapshots U solving K U = F.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class IndustrialCase:
    name: str
    operator: np.ndarray
    train_forces: np.ndarray
    train_states: np.ndarray
    test_forces: np.ndarray
    test_states: np.ndarray
    direct_solve_time_per_case_sec: float
    metadata: dict


def _solve_many(K: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, float]:
    import time
    start = time.perf_counter()
    U = np.linalg.solve(K, F)
    elapsed = time.perf_counter() - start
    return U, elapsed / max(1, F.shape[1])


def _normalize_states(U_train: np.ndarray, U_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    scale = np.linalg.norm(U_train, ord="fro") / np.sqrt(max(1, U_train.size))
    if scale <= 0 or not np.isfinite(scale):
        scale = 1.0
    return U_train / scale, U_test / scale, scale


def build_truss_bridge_case(n_train: int, n_test: int, seed: int, n_panels: int = 8) -> IndustrialCase:
    """Two-dimensional steel truss bridge segment with moving deck loads and wind."""
    rng = np.random.default_rng(seed)
    span, height = 30.0, 4.0
    E = 200e9
    A_chord, A_vert, A_diag = 4.0e-3, 2.5e-3, 2.0e-3
    xs = np.linspace(0.0, span, n_panels + 1)
    bottom = np.column_stack([xs, np.zeros_like(xs)])
    top = np.column_stack([xs, height * np.ones_like(xs)])
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

    def element_k(xi: np.ndarray, xj: np.ndarray, area: float) -> np.ndarray:
        dx, dy = xj - xi
        L = float(np.sqrt(dx * dx + dy * dy))
        c, s = dx / L, dy / L
        return (E * area / L) * np.array([
            [ c*c,  c*s, -c*c, -c*s],
            [ c*s,  s*s, -c*s, -s*s],
            [-c*c, -c*s,  c*c,  c*s],
            [-c*s, -s*s,  c*s,  s*s],
        ])

    ndof = 2 * n_nodes
    K_full = np.zeros((ndof, ndof), dtype=float)
    for ni, nj, area, _kind in elements:
        ke = element_k(nodes[ni], nodes[nj], area)
        dofs = np.array([2 * ni, 2 * ni + 1, 2 * nj, 2 * nj + 1])
        K_full[np.ix_(dofs, dofs)] += ke
    fixed = np.array([0, 1, 2 * n_panels + 1], dtype=int)
    free = np.setdiff1d(np.arange(ndof), fixed)
    K = K_full[np.ix_(free, free)]

    def loads(n_cases: int, offset: int) -> np.ndarray:
        F_full = np.zeros((ndof, n_cases), dtype=float)
        bottom_nodes = np.arange(1, n_panels)
        top_nodes = n_bottom + np.arange(1, n_panels)
        t = np.linspace(0.0, 1.0, n_cases)
        for j in range(n_cases):
            fj = np.zeros(ndof)
            deck = 8e4 * (0.8 + 0.25 * np.sin(2 * np.pi * t[j]) + 0.08 * rng.normal())
            truck = 9e4 * (1.0 + 0.15 * rng.normal())
            wind = 2e4 * (0.5 + 0.5 * np.sin(2 * np.pi * (t[j] + 0.15 + 0.03 * offset)))
            for node in bottom_nodes:
                fj[2 * node + 1] += -deck / max(1, len(bottom_nodes))
            moving_node = int(bottom_nodes[int(np.round(t[j] * (len(bottom_nodes) - 1)))])
            fj[2 * moving_node + 1] += -truck
            for node in top_nodes:
                fj[2 * node] += wind / max(1, len(top_nodes))
            if j % 7 == 0:
                node = int(rng.choice(top_nodes))
                fj[2 * node + 1] += -0.25 * truck
            F_full[:, j] = fj
        return F_full[free, :]

    F_train = loads(n_train, 0)
    F_test = loads(n_test, 1)
    U_train, t_train = _solve_many(K, F_train)
    U_test, t_test = _solve_many(K, F_test)
    U_train, U_test, scale = _normalize_states(U_train, U_test)
    return IndustrialCase(
        "structural_truss_bridge",
        K,
        F_train / scale,
        U_train,
        F_test / scale,
        U_test,
        0.5 * (t_train + t_test),
        {"description": "steel truss bridge FE reduced free-DOF model", "n_free_dof": K.shape[0], "n_elements": len(elements)},
    )


def build_thermal_process_case(n_train: int, n_test: int, seed: int, n_nodes: int = 90) -> IndustrialCase:
    """Steady heat conduction in an insulated industrial bar/process line with variable heaters."""
    rng = np.random.default_rng(10_000 + seed)
    n = n_nodes
    dx = 1.0 / (n + 1)
    conductivity = 16.0
    leakage = 0.8
    main = (2.0 * conductivity / dx**2 + leakage) * np.ones(n)
    off = (-conductivity / dx**2) * np.ones(n - 1)
    K = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)
    x = np.linspace(dx, 1.0 - dx, n)
    heater_centers = np.array([0.18, 0.43, 0.72, 0.88])
    heater_width = 0.045
    profiles = np.vstack([np.exp(-0.5 * ((x - c) / heater_width) ** 2) for c in heater_centers]).T
    ambient = np.ones((n, 1))

    def forces(n_cases: int, phase: float) -> np.ndarray:
        t = np.linspace(0.0, 1.0, n_cases)
        F = np.zeros((n, n_cases))
        for j in range(n_cases):
            coeff = np.array([
                40 + 12 * np.sin(2 * np.pi * (t[j] + phase)),
                25 + 8 * np.cos(2 * np.pi * (1.7 * t[j] + phase)),
                35 + 10 * np.sin(2 * np.pi * (0.7 * t[j] + 0.2)),
                15 + 5 * rng.normal(),
            ])
            F[:, j] = profiles @ coeff + 2.0 * ambient[:, 0] + rng.normal(0.0, 0.3, size=n)
        return F

    F_train = forces(n_train, 0.0)
    F_test = forces(n_test, 0.15)
    U_train, t_train = _solve_many(K, F_train)
    U_test, t_test = _solve_many(K, F_test)
    U_train, U_test, scale = _normalize_states(U_train, U_test)
    return IndustrialCase(
        "thermal_process_line",
        K,
        F_train / scale,
        U_train,
        F_test / scale,
        U_test,
        0.5 * (t_train + t_test),
        {"description": "steady heat equation with localized heaters and leakage", "n_state": n},
    )


def build_dc_grid_case(n_train: int, n_test: int, seed: int, n_buses: int = 85) -> IndustrialCase:
    """DC power-flow surrogate with changing loads, renewables, and one slack bus removed."""
    rng = np.random.default_rng(20_000 + seed)
    n = n_buses
    W = np.zeros((n, n))
    for i in range(n - 1):
        b = rng.uniform(3.0, 10.0)
        W[i, i + 1] = W[i + 1, i] = b
    for _ in range(n // 3):
        i, j = rng.choice(n, size=2, replace=False)
        if abs(i - j) > 2:
            b = rng.uniform(0.5, 3.0)
            W[i, j] = W[j, i] = b
    L = np.diag(W.sum(axis=1)) - W
    # Remove slack bus and add a tiny reference regularization.
    K = L[1:, 1:] + 1e-6 * np.eye(n - 1)
    m = n - 1
    industrial_loads = rng.choice(np.arange(m), size=max(4, m // 8), replace=False)
    renewable_buses = rng.choice(np.setdiff1d(np.arange(m), industrial_loads), size=max(4, m // 10), replace=False)

    def injections(n_cases: int, phase: float) -> np.ndarray:
        t = np.linspace(0.0, 1.0, n_cases)
        P = np.zeros((m, n_cases))
        for j in range(n_cases):
            base = -0.25 - 0.08 * np.sin(2 * np.pi * (t[j] + phase))
            p = base * np.ones(m) + 0.03 * rng.normal(size=m)
            p[industrial_loads] += -0.8 * (0.7 + 0.3 * np.sin(2 * np.pi * (2 * t[j] + phase)))
            solar = 0.9 * max(0.0, np.sin(np.pi * t[j]))
            wind = 0.4 + 0.2 * np.sin(2 * np.pi * (3 * t[j] + 0.2))
            p[renewable_buses] += solar + wind + 0.05 * rng.normal(size=len(renewable_buses))
            # Slack absorbs imbalance; reduced system sees the non-slack injections.
            P[:, j] = p
        return P

    F_train = injections(n_train, 0.0)
    F_test = injections(n_test, 0.18)
    U_train, t_train = _solve_many(K, F_train)
    U_test, t_test = _solve_many(K, F_test)
    U_train, U_test, scale = _normalize_states(U_train, U_test)
    return IndustrialCase(
        "dc_power_grid_loadflow",
        K,
        F_train / scale,
        U_train,
        F_test / scale,
        U_test,
        0.5 * (t_train + t_test),
        {"description": "DC load-flow angles on a meshed transmission-like grid", "n_non_slack_buses": m},
    )


def build_all_cases(n_train: int, n_test: int, seed: int) -> list[IndustrialCase]:
    return [
        build_truss_bridge_case(n_train, n_test, seed),
        build_thermal_process_case(n_train, n_test, seed),
        build_dc_grid_case(n_train, n_test, seed),
    ]


def add_measurement_noise(U: np.ndarray, noise_level: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(90_000 + seed)
    rms = np.linalg.norm(U, ord="fro") / np.sqrt(max(1, U.size))
    return U + noise_level * rms * rng.normal(size=U.shape)
