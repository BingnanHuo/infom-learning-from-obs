# OGBench Cube-Single Reproduction A/B Submissions

Date: 2026-05-05

## Purpose

Diagnose whether task 1 seed 0 underperformance is caused by current feature-branch code paths or by upstream/data/environment behavior.

All jobs use OGBench cube-single task 1, seed 0, paper hyperparameters, the same Unity Python environment, and the same `~/.ogbench/data` dataset files.

## Submitted Jobs

| Pair | Variant | Job ID | Partition | Constraint | Run root |
| --- | --- | ---: | --- | --- | --- |
| H100 | upstream clean `ed0761d` | 56723248 | gpu-preempt | h100 | `/work/pi_mengfanxu_umass_edu/bhuo_umass_edu/infom-learning-from-obs/runs/unity_cube_single_repro_ab_20260505_h100_t1s0` |
| H100 | current feature, no checkpoint/TensorBoard extras | 56723249 | gpu-preempt | h100 | `/work/pi_mengfanxu_umass_edu/bhuo_umass_edu/infom-learning-from-obs/runs/unity_cube_single_repro_ab_20260505_h100_t1s0` |
| A100 | upstream clean `ed0761d` | 56723253 | gpu-preempt | a100 | `/work/pi_mengfanxu_umass_edu/bhuo_umass_edu/infom-learning-from-obs/runs/unity_cube_single_repro_ab_20260505_a100_t1s0` |
| A100 | current feature, no checkpoint/TensorBoard extras | 56723254 | gpu-preempt | a100 | `/work/pi_mengfanxu_umass_edu/bhuo_umass_edu/infom-learning-from-obs/runs/unity_cube_single_repro_ab_20260505_a100_t1s0` |

## Interpretation Rules

- If upstream and current-no-extras agree on the same GPU class, current bridge/checkpoint changes are unlikely to be the main cause.
- If current-no-extras is worse than upstream on the same GPU class, inspect current `main.py`, `envs/env_utils.py`, and `agents/infom.py` diffs before running more seeds.
- If both upstream and current are below paper on both GPU classes, prioritize dependency/version and fine-tuning dataset fidelity.
- If A100 and H100 differ materially for the same variant, treat hardware/JAX/XLA nondeterminism as a first-class reproduction risk.
- Paper metric is success averaged over steps 1,400,000, 1,450,000, and 1,500,000; best success is only diagnostic.

## Current Evidence Before A/B Completion

- Local task 1 seed 0 checkpointed feature run: 21.33% paper-window success, 18.00% final success.
- Unity task 1 seed 0 checkpointed feature run: 21.33% paper-window success, 0.00% final success, with early success up to 94% and late collapse.
- Unity upstream task 1 seeds 0-3: 41.33 +/- 32.43% paper-window success, below the paper target of 92.5 +/- 4.0%.
- Unity task 4 and task 5 seed 0 feature runs are close to expected paper ranges, so the issue is not a blanket inability to learn cube-single.
