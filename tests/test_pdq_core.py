import numpy as np

from pdqbench.cases import build_thermal_process_case
from pdqbench.metrics import rel_fro_error
from pdqbench.pdq import fit_all_methods
from pdqbench.sensors import reconstruct_from_sensors, select_variance_sensors


def test_thermal_case_shapes():
    case = build_thermal_process_case(n_train=10, n_test=4, seed=13, n_nodes=20)
    assert case.operator.shape == (20, 20)
    assert case.train_states.shape == (20, 10)
    assert case.test_states.shape == (20, 4)
    assert np.all(np.linalg.eigvalsh(case.operator) > 0)


def test_pdq_methods_fit_without_nan():
    rng = np.random.default_rng(13)
    A = rng.normal(size=(25, 14))
    cfg = {"max_sweeps": 5, "kappa_core_max": 20.0}
    methods, histories = fit_all_methods(A, r=4, seed=13, cfg=cfg)
    names = [m.method for m in methods]
    assert "Truncated SVD / POD" in names
    assert "PDQ conditioned core" in names
    assert histories["PDQ conditioned core"]["kappa_D"].iloc[-1] <= 20.0 + 1e-8
    for model in methods:
        Ahat = model.reconstruct_full(A)
        assert np.isfinite(rel_fro_error(A, Ahat))


def test_sparse_sensor_reconstruction_runs():
    case = build_thermal_process_case(n_train=12, n_test=3, seed=29, n_nodes=24)
    cfg = {"max_sweeps": 4, "kappa_core_max": 50.0}
    methods, _ = fit_all_methods(case.train_states, r=3, seed=29, cfg=cfg)
    rows = select_variance_sensors(case.train_states, sensor_count=5)
    model = [m for m in methods if m.method == "PDQ conditioned core"][0]
    y = case.test_states[rows, 0]
    uhat = reconstruct_from_sensors(model.mean, model.basis, rows, y, ridge=1e-8)
    assert uhat.shape == (case.test_states.shape[0],)
    assert np.all(np.isfinite(uhat))
