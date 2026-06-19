# Scientific scope

This repository validates core-conditioned PDQ factorization as a low-rank surrogate method for industrial response fields.

The correct scientific claim is:

> PDQ with ridge-regularized side factors and conditioned cores can be evaluated as a stable low-rank surrogate for structured operator-response data, especially when sparse-sensor reconstruction and deployment metrics matter.

The repository does **not** claim:

- superiority on every low-rank approximation problem,
- proprietary industrial deployment,
- replacement of full FE/thermal/grid solvers in certified safety-critical settings,
- exact H-matrix implementation.

## Evidence expected from a serious run

A strong run should report:

- reconstruction error on clean test responses,
- reconstruction error under noisy training snapshots,
- sparse-sensor recovery error,
- physical energy-norm error,
- operator residual,
- runtime and speed-up,
- storage ratio,
- core condition number,
- ablation evidence showing why side regularization, core conditioning, and sensor placement matter.
