# GitHub release checklist

Before making the repository public:

1. Run `pytest -q`.
2. Run `python scripts/run_multi_industry_pdq_benchmarks.py --smoke`.
3. Run at least one full benchmark grid.
4. Check `main_table.csv` and `figures/`.
5. Do not delete `legacy/`; it preserves provenance.
6. Put representative results in the README only after verifying them on your machine.
7. Use cautious language: "industry-facing" and "industry-realistic" unless using public real matrices.
8. Add a GitHub release tag after the first reproducible result folder is archived.
