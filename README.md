# PDQ Industrial Surrogate Benchmarks

> **Short description:** Core-conditioned PDQ benchmarks for industrial reduced-order models, digital twins, and sparse-sensor reconstruction.
Industry-facing validation suite for **core-conditioned PDQ factorization** in reduced-order modelling, digital-twin compression, and sparse-sensor reconstruction.

The repository turns exploratory PDQ validation scripts into a clean, testable, GitHub-ready scientific-computing package. It includes real and industry-realistic validation settings, strong baselines, ablations, physical error metrics, and reproducible output tables.

## What is being validated?

Given a response/snapshot matrix

```text
A = [u_1, u_2, ..., u_n],
```

PDQ approximates the centered response field as

```text
A ≈ mean(A) + P D Q,
```

where `P` and `Q` are side factors and `D` is a small core matrix. The main variant controls the conditioning of `D`, which is important for stable deployment and sparse-sensor reconstruction.

## Repository layout

```text
pdq-industrial-surrogate-benchmarks/
├── src/pdqbench/                       # reusable package
│   ├── pdq.py                          # PDQ, SVD, randomized SVD, ablations
│   ├── cases.py                        # structural, thermal, and grid benchmark cases
│   ├── sensors.py                      # sensor selection and reconstruction
│   ├── metrics.py                      # physical and numerical metrics
│   └── reporting.py                    # aggregation and figures
├── scripts/
│   └── run_multi_industry_pdq_benchmarks.py
├── legacy/                             # original uploaded research scripts, preserved
├── tests/                              # lightweight CI tests
├── docs/                               # audit, scientific scope, release checklist
└── .github/workflows/ci.yml
```

## Main benchmark script

The new main script is:

```bash
python scripts/run_multi_industry_pdq_benchmarks.py
```

It runs three deployment-style case studies:

| Case | Industrial meaning | Operator solved |
|---|---|---|
| `structural_truss_bridge` | FE bridge/digital-twin displacement recovery | structural stiffness system |
| `thermal_process_line` | process-line temperature field recovery | steady heat-conduction system |
| `dc_power_grid_loadflow` | grid angle/state recovery from sparse measurements | reduced DC load-flow system |

The script evaluates:

- full snapshot reconstruction,
- sparse-sensor reconstruction,
- energy-norm error,
- operator/equilibrium residual,
- max-response error,
- offline fitting time,
- online reconstruction time,
- speed-up against direct solves,
- storage ratio,
- core condition number.

## Baselines and ablations

The main script compares:

1. Mean response baseline.
2. Truncated SVD / POD.
3. Randomized SVD.
4. PDQ ridge.
5. PDQ conditioned core.
6. PDQ weak-side-ridge ablation.
7. PDQ random-initialization ablation.
8. Variance-based sensors versus random-sensor ablation.

This makes the evidence harder to dismiss because the proposed method is not compared only against weak baselines.

## Installation

Using Anaconda Prompt:

```bat
conda create -n pdqbench python=3.10 -y
conda activate pdqbench
pip install -r requirements.txt
pip install -e .
pytest -q
```

## Smoke test

```bat
python scripts\run_multi_industry_pdq_benchmarks.py --smoke
```

This produces a small timestamped results folder with CSV tables, LaTeX tables, convergence histories, and figures.

## Full run

```bat
python scripts\run_multi_industry_pdq_benchmarks.py ^
  --seeds 13 29 47 ^
  --ranks 4 8 12 ^
  --sensor-counts 6 10 16 ^
  --training-noise 0.00 0.01 0.03 ^
  --sensor-noise 0.00 0.01 0.03 ^
  --n-train 80 ^
  --n-test 40
```

On Linux/macOS:

```bash
python scripts/run_multi_industry_pdq_benchmarks.py \
  --seeds 13 29 47 \
  --ranks 4 8 12 \
  --sensor-counts 6 10 16 \
  --training-noise 0.00 0.01 0.03 \
  --sensor-noise 0.00 0.01 0.03 \
  --n-train 80 \
  --n-test 40
```

## Outputs

A folder such as `results_multi_industry_pdq_YYYYMMDD_HHMMSS/` is created with:

```text
config.json
case_metadata.csv
all_runs.csv
aggregate_results.csv
main_table.csv
main_table.tex
histories/*.csv
figures/*.png
```

## How to position this repository

Use careful language:

- Good: **industry-facing**, **industrial surrogate benchmark**, **structural/thermal/grid digital-twin validation**, **public structural matrix validation preserved in legacy scripts**.
- Avoid: claiming proprietary industrial deployment unless you have company data.
- Avoid: saying the synthetic thermal and grid cases are “real data.” They are physically structured industrial case studies.

## Original scripts preserved

The original uploaded files are kept under `legacy/` for traceability:

- `industrial_pdq_validation_v2_original.py`
- `real_industrial_pdq_validation_original.py`
- `cas_structural_pdq_validation_original.py`

The cleaned package and `scripts/run_multi_industry_pdq_benchmarks.py` are the preferred public entry points.
