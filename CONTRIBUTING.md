# Contributing

This repository is designed as reproducible scientific-computing research code.

Before submitting changes:

1. Keep scripts deterministic through explicit seeds.
2. Do not add machine-specific paths.
3. Do not report only a single favourable run; use grids and aggregate tables.
4. Add a baseline or ablation when introducing a new proposed variant.
5. Run:

```bash
python -m compileall src scripts tests
pytest -q
```

For new industrial cases, include:

- the physical/operator model,
- the scenario generator,
- the measured outputs,
- at least one physical error metric,
- a sparse-sensor setting,
- a fair baseline comparison.
