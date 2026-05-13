# OGBench Cube-Single InFOM Paper-Reproduction Diagnosis

- Generated at: `2026-05-05T18:27:12.086624+00:00`
- Roots scanned: 1
- Runs parsed: 1

## Environment

- Python: `3.10.16 (main, Dec 11 2024, 16:24:50) [GCC 11.2.0]`
- Repo: `/home/nick/infom-learning-from-obs`
- Git head: `04cbbf64a263b6bd92bece81827677c2e49b6115`
- Git status: `## feature/tensorboard-local-ogbench-runs...origin/feature/tensorboard-local-ogbench-runs;  M main.py;  M utils/flax_utils.py;  M utils/log_utils.py; ?? runs/reports/ogbench_paper_repro/; ?? scripts/diagnose_ogbench_paper_repro.py; ?? scripts/jax_cuda_probe.py; ?? scripts/submit_ogbench_cube_single_repro_unity.py; ?? scripts/submit_ogbench_repro_ab_unity.py; ?? scripts/summarize_ogbench_paper_repro.py`
- Packages: `{"distrax": "0.1.5", "dm_control": "1.0.39", "flax": "0.10.7", "gymnasium": "0.29.1", "jax": "0.6.2", "jaxlib": "0.6.2", "ml_collections": "1.1.0", "mujoco": "3.7.0", "numpy": "2.2.6", "ogbench": "1.1.0", "optax": "0.2.8"}`

### Dataset Hashes

| File | Bytes | SHA256 |
| --- | ---: | --- |
| cube-single-play-v0.npz | 256728298 | `80f3b6fd27f4f9d9e9eb6f0d07d6951559012f45b1e15ea4046ef8ecd8d3684e` |
| cube-single-play-v0-val.npz | 25687020 | `96d07401bdebdc3f0ea6d56ed1333863e0962f441483adc2c43b83105046eb00` |
| cube-single-play-ft-v0.npz | 126403421 | `f91817a1aba8f379676ef1aa8b24d8fd5e1bf4ca527301327a50a9b1fc38ae07` |
| cube-single-play-ft-v0-val.npz | 12646389 | `efe81dd66aea2952ebc942d98b2f303540ab18a7879033a724cd80687d410b1e` |

## Paper Metric

| Root | Task | Seeds | Paper success % mean +/- sd | Paper target % | Delta vs target | Final success % mean +/- sd |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| local_task1 | 1 | 1 | 21.33 +/- 0.00 | 92.5 +/- 4.0 | -71.17 | 18.00 +/- 0.00 |

## Run Health

| Root | Task | Seed | Best % | Paper % | Final % | Grad norm | Val critic loss | Actor MSE | Health flags | Run dir |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| local_task1 | 1 | 0 | 30.00 | 21.33 | 18.00 | 40.639 [min 0.379, max 9276.553] | 1797605.600 [min 1.136, max 20927688.000] | 0.010 [min 0.009, max 0.023] | training grad norm spike 9.28e+03; validation critic loss spike 2.09e+07; validation flow loss spike 3.84e+06; paper-window success far below target by 71.2 pp | `/home/nick/infom-runs/local_cube_single_priority_20260505_0005/runs/local_cube_single_task1/task1_seed0` |

## Protocol Checks

| Root | Task | Seed | Protocol issues | Command-shape extras |
| --- | ---: | ---: | --- | --- |
| local_task1 | 1 | 0 | none | enable_tensorboard=1; checkpoint_interval=50000; checkpoint_at_end=1; resume_from_checkpoint=1 |

## Eval Curves

- `local_task1` task 1 seed 0: 1000001:0, 1050000:2, 1100000:18, 1150000:4, 1200000:12, 1250000:0, 1300000:16, 1350000:6, 1400000:30, 1450000:16, 1500000:18

## Slurm And Errors

- `local_task1`: 0 Slurm log summaries, 0 error-pattern hits.

## Notes

- Paper metric here is mean success at 1.4M, 1.45M, and 1.5M total steps.
- Best success is diagnostic only and is not the paper metric.
- Validation spikes are flagged as diagnostics; they are not by themselves proof of failed policy learning.
