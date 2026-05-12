# OGBench Cube-Single InFOM Paper-Reproduction Diagnosis

- Generated at: `2026-05-05T18:27:47.202451+00:00`
- Roots scanned: 2
- Runs parsed: 10

## Environment

- Python: `3.10.16 | packaged by conda-forge | (main, Apr  8 2025, 20:53:32) [GCC 13.3.0]`
- Repo: `/work/pi_mengfanxu_umass_edu/bhuo_umass_edu/repos/infom-learning-from-obs`
- Git head: `04cbbf64a263b6bd92bece81827677c2e49b6115`
- Git status: `## feature/tensorboard-local-ogbench-runs...origin/feature/tensorboard-local-ogbench-runs;  M main.py;  M utils/flax_utils.py;  M utils/log_utils.py; ?? runs/reports/ogbench_paper_repro/; ?? scripts/diagnose_ogbench_paper_repro.py; ?? scripts/jax_cuda_probe.py; ?? scripts/submit_ogbench_cube_single_repro_unity.py; ?? scripts/submit_ogbench_repro_ab_unity.py; ?? scripts/summarize_ogbench_paper_repro.py`
- Packages: `{"distrax": "0.1.5", "dm_control": "1.0.40", "flax": "0.10.7", "gymnasium": "0.29.1", "jax": "0.6.2", "jaxlib": "0.6.2", "ml_collections": "1.1.0", "mujoco": "3.8.0", "numpy": "2.2.6", "ogbench": "1.1.0", "optax": "0.2.8"}`

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
| unity_checkpointed_seed0 | 1 | 1 | 21.33 +/- 0.00 | 92.5 +/- 4.0 | -71.17 | 0.00 +/- 0.00 |
| unity_checkpointed_seed0 | 2 | 1 | 19.33 +/- 0.00 | 78.4 +/- 12.3 | -59.07 | 8.00 +/- 0.00 |
| unity_checkpointed_seed0 | 3 | 1 | 61.33 +/- 0.00 | 56.4 +/- 36.9 | 4.93 | 0.00 +/- 0.00 |
| unity_checkpointed_seed0 | 4 | 1 | 82.00 +/- 0.00 | 91.5 +/- 14.2 | -9.50 | 84.00 +/- 0.00 |
| unity_checkpointed_seed0 | 5 | 1 | 74.67 +/- 0.00 | 70.0 +/- 39.1 | 4.67 | 86.00 +/- 0.00 |
| unity_upstream_task1_4seed | 1 | 4 | 41.33 +/- 32.43 | 92.5 +/- 4.0 | -51.17 | 34.00 +/- 38.19 |

## Run Health

| Root | Task | Seed | Best % | Paper % | Final % | Grad norm | Val critic loss | Actor MSE | Health flags | Run dir |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| unity_checkpointed_seed0 | 1 | 0 | 94.00 | 21.33 | 0.00 | 1437.009 [min 0.372, max 4730.934] | 1262329.100 [min 1.783, max 134015000.000] | 0.519 [min 0.011, max 0.682] | training grad norm spike 4.73e+03; validation critic loss spike 1.34e+08; validation flow loss spike 5.87e+05; training actor MSE spike 0.682; training actor q_mean positive spike 983; success collapse best 0.94 to final 0.00; paper-window success far below target by 71.2 pp | `/work/pi_mengfanxu_umass_edu/bhuo_umass_edu/infom-learning-from-obs/runs/unity_checkpoint_smoke_20260505_0002/runs/unity_cube_single_paper_repro_task1/task1_seed0` |
| unity_checkpointed_seed0 | 1 | 0 | 0.00 | NA | 0.00 | 90.059 [min 2.035, max 158.648] | 196.113 [min 196.113, max 237.145] | NA | none | `/work/pi_mengfanxu_umass_edu/bhuo_umass_edu/infom-learning-from-obs/runs/unity_checkpoint_smoke_20260505_0002/smoke_runs/unity_cube_single_paper_repro_smoke/smoke` |
| unity_checkpointed_seed0 | 2 | 0 | 84.00 | 19.33 | 8.00 | 449.921 [min 0.379, max 566.625] | 3407544.200 [min 2.050, max 34804224.000] | 0.155 [min 0.014, max 0.205] | validation critic loss spike 3.48e+07; validation flow loss spike 2.92e+05; training actor MSE spike 0.205; success collapse best 0.84 to final 0.08; paper-window success far below target by 59.1 pp | `/work/pi_mengfanxu_umass_edu/bhuo_umass_edu/infom-learning-from-obs/runs/unity_checkpoint_smoke_20260505_0002/runs/unity_cube_single_paper_repro_task2/task2_seed0` |
| unity_checkpointed_seed0 | 3 | 0 | 100.00 | 61.33 | 0.00 | 983.771 [min 0.379, max 983.771] | 932201.000 [min 2.154, max 27903796.000] | 0.408 [min 0.014, max 0.435] | validation critic loss spike 2.79e+07; validation flow loss spike 2.92e+05; training actor MSE spike 0.435; success collapse best 1.00 to final 0.00 | `/work/pi_mengfanxu_umass_edu/bhuo_umass_edu/infom-learning-from-obs/runs/unity_checkpoint_smoke_20260505_0002/runs/unity_cube_single_paper_repro_task3/task3_seed0` |
| unity_checkpointed_seed0 | 4 | 0 | 98.00 | 82.00 | 84.00 | 55.141 [min 0.379, max 103.308] | 1519196.900 [min 1.831, max 15171264.000] | 0.017 [min 0.014, max 0.026] | validation critic loss spike 1.52e+07; validation flow loss spike 2.2e+06 | `/work/pi_mengfanxu_umass_edu/bhuo_umass_edu/infom-learning-from-obs/runs/unity_checkpoint_smoke_20260505_0002/runs/unity_cube_single_paper_repro_task4/task4_seed0` |
| unity_checkpointed_seed0 | 5 | 0 | 96.00 | 74.67 | 86.00 | 41.718 [min 0.379, max 73.273] | 20405.357 [min 1.663, max 127977.810] | 0.017 [min 0.013, max 0.022] | validation flow loss spike 1.67e+06 | `/work/pi_mengfanxu_umass_edu/bhuo_umass_edu/infom-learning-from-obs/runs/unity_checkpoint_smoke_20260505_0002/runs/unity_cube_single_paper_repro_task5/task5_seed0` |
| unity_upstream_task1_4seed | 1 | 0 | 90.00 | 64.67 | 64.00 | 42.744 [min 0.379, max 74.217] | 263654.250 [min 2.469, max 4107210.800] | 0.023 [min 0.014, max 0.026] | validation critic loss spike 4.11e+06; validation flow loss spike 2.86e+05; paper-window success far below target by 27.8 pp | `/work/pi_mengfanxu_umass_edu/bhuo_umass_edu/infom-learning-from-obs/runs/unity_cube_single_paper_repro_retry_exclude_gpu026_20260504_2236/runs/unity_cube_single_paper_repro_task1/sd000_s_56693855.0.20260504_223853` |
| unity_upstream_task1_4seed | 1 | 1 | 16.00 | 2.67 | 0.00 | 30.372 [min 0.371, max 357.587] | 4.270 [min 0.162, max 2193.435] | 0.012 [min 0.010, max 0.025] | paper-window success far below target by 89.8 pp | `/work/pi_mengfanxu_umass_edu/bhuo_umass_edu/infom-learning-from-obs/runs/unity_cube_single_paper_repro_retry_exclude_gpu026_20260504_2236/runs/unity_cube_single_paper_repro_task1/sd001_s_56693856.0.20260504_223748` |
| unity_upstream_task1_4seed | 1 | 2 | 96.00 | 26.67 | 2.00 | 458.876 [min 0.322, max 928.078] | 78939.125 [min 2.489, max 141203200.000] | 0.223 [min 0.012, max 0.232] | validation critic loss spike 1.41e+08; training actor MSE spike 0.232; success collapse best 0.96 to final 0.02; paper-window success far below target by 65.8 pp | `/work/pi_mengfanxu_umass_edu/bhuo_umass_edu/infom-learning-from-obs/runs/unity_cube_single_paper_repro_retry_exclude_gpu026_20260504_2236/runs/unity_cube_single_paper_repro_task1/sd002_s_56693857.0.20260504_223753` |
| unity_upstream_task1_4seed | 1 | 3 | 100.00 | 71.33 | 70.00 | 27.760 [min 0.324, max 130.789] | 51.314 [min 1.417, max 372028.300] | 0.019 [min 0.013, max 0.027] | validation flow loss spike 3.2e+05 | `/work/pi_mengfanxu_umass_edu/bhuo_umass_edu/infom-learning-from-obs/runs/unity_cube_single_paper_repro_retry_exclude_gpu026_20260504_2236/runs/unity_cube_single_paper_repro_task1/sd003_s_56693858.0.20260504_224004` |

## Protocol Checks

| Root | Task | Seed | Protocol issues | Command-shape extras |
| --- | ---: | ---: | --- | --- |
| unity_checkpointed_seed0 | 1 | 0 | none | enable_tensorboard=1; checkpoint_interval=50000; checkpoint_at_end=1; resume_from_checkpoint=1 |
| unity_checkpointed_seed0 | 1 | 0 | pretraining_steps=2 expected 1000000; finetuning_steps=2 expected 500000; eval_interval=1 expected 50000; eval_episodes=1 expected 50 | enable_tensorboard=1; checkpoint_interval=2; checkpoint_at_end=1; resume_from_checkpoint=1 |
| unity_checkpointed_seed0 | 2 | 0 | none | enable_tensorboard=1; checkpoint_interval=50000; checkpoint_at_end=1; resume_from_checkpoint=1 |
| unity_checkpointed_seed0 | 3 | 0 | none | enable_tensorboard=1; checkpoint_interval=50000; checkpoint_at_end=1; resume_from_checkpoint=1 |
| unity_checkpointed_seed0 | 4 | 0 | none | enable_tensorboard=1; checkpoint_interval=50000; checkpoint_at_end=1; resume_from_checkpoint=1 |
| unity_checkpointed_seed0 | 5 | 0 | none | enable_tensorboard=1; checkpoint_interval=50000; checkpoint_at_end=1; resume_from_checkpoint=1 |
| unity_upstream_task1_4seed | 1 | 0 | none | none |
| unity_upstream_task1_4seed | 1 | 1 | none | none |
| unity_upstream_task1_4seed | 1 | 2 | none | none |
| unity_upstream_task1_4seed | 1 | 3 | none | none |

## Eval Curves

- `unity_checkpointed_seed0` task 1 seed 0: 1000001:0, 1050000:0, 1100000:94, 1150000:90, 1200000:90, 1250000:78, 1300000:60, 1350000:2, 1400000:64, 1450000:0, 1500000:0
- `unity_checkpointed_seed0` task 1 seed 0: 3:0, 4:0
- `unity_checkpointed_seed0` task 2 seed 0: 1000001:0, 1050000:10, 1100000:72, 1150000:84, 1200000:64, 1250000:32, 1300000:56, 1350000:22, 1400000:44, 1450000:6, 1500000:8
- `unity_checkpointed_seed0` task 3 seed 0: 1000001:0, 1050000:28, 1100000:94, 1150000:94, 1200000:100, 1250000:96, 1300000:96, 1350000:46, 1400000:92, 1450000:92, 1500000:0
- `unity_checkpointed_seed0` task 4 seed 0: 1000001:0, 1050000:0, 1100000:82, 1150000:84, 1200000:88, 1250000:98, 1300000:94, 1350000:84, 1400000:82, 1450000:80, 1500000:84
- `unity_checkpointed_seed0` task 5 seed 0: 1000001:0, 1050000:0, 1100000:18, 1150000:84, 1200000:80, 1250000:82, 1300000:96, 1350000:68, 1400000:66, 1450000:72, 1500000:86
- `unity_upstream_task1_4seed` task 1 seed 0: 1000001:0, 1050000:0, 1100000:58, 1150000:72, 1200000:64, 1250000:90, 1300000:62, 1350000:60, 1400000:44, 1450000:86, 1500000:64
- `unity_upstream_task1_4seed` task 1 seed 1: 1000001:0, 1050000:0, 1100000:0, 1150000:8, 1200000:10, 1250000:2, 1300000:16, 1350000:12, 1400000:6, 1450000:2, 1500000:0
- `unity_upstream_task1_4seed` task 1 seed 2: 1000001:0, 1050000:2, 1100000:8, 1150000:96, 1200000:90, 1250000:78, 1300000:94, 1350000:92, 1400000:70, 1450000:8, 1500000:2
- `unity_upstream_task1_4seed` task 1 seed 3: 1000001:0, 1050000:48, 1100000:76, 1150000:88, 1200000:92, 1250000:100, 1300000:60, 1350000:76, 1400000:76, 1450000:68, 1500000:70

## Slurm And Errors

- `unity_checkpointed_seed0`: 6 Slurm log summaries, 0 error-pattern hits.
  - job `56703078` host `gpu013` repo `04cbbf64a263b6bd92bece81827677c2e49b6115` driver `590.48.01` cuda `13.1`
  - job `56703080` host `uri-gpu012` repo `04cbbf64a263b6bd92bece81827677c2e49b6115` driver `580.126.09` cuda `13.0`
  - job `56704942` host `uri-gpu004` repo `04cbbf64a263b6bd92bece81827677c2e49b6115` driver `580.126.09` cuda `13.0`
  - job `56704943` host `uri-gpu004` repo `04cbbf64a263b6bd92bece81827677c2e49b6115` driver `580.126.09` cuda `13.0`
  - job `56704944` host `uri-gpu012` repo `04cbbf64a263b6bd92bece81827677c2e49b6115` driver `580.126.09` cuda `13.0`
  - job `56704945` host `uri-gpu004` repo `04cbbf64a263b6bd92bece81827677c2e49b6115` driver `580.126.09` cuda `13.0`
- `unity_upstream_task1_4seed`: 20 Slurm log summaries, 80 error-pattern hits.
  - job `56693855` host `gpu027` repo `ed0761d7a349fb34b201071f98ac88b6d91cafe2` driver `580.126.09` cuda `13.0`
  - job `56693856` host `gpu032` repo `ed0761d7a349fb34b201071f98ac88b6d91cafe2` driver `580.126.09` cuda `13.0`
  - job `56693857` host `gpu029` repo `ed0761d7a349fb34b201071f98ac88b6d91cafe2` driver `580.126.09` cuda `13.0`
  - job `56693858` host `gpu030` repo `ed0761d7a349fb34b201071f98ac88b6d91cafe2` driver `580.126.09` cuda `13.0`
  - job `56693859` host `gpu030` repo `ed0761d7a349fb34b201071f98ac88b6d91cafe2` driver `580.126.09` cuda `13.0`
  - job `56693860` host `gpu030` repo `ed0761d7a349fb34b201071f98ac88b6d91cafe2` driver `580.126.09` cuda `13.0`
  - job `56693861` host `gpu030` repo `ed0761d7a349fb34b201071f98ac88b6d91cafe2` driver `580.126.09` cuda `13.0`
  - job `56693862` host `gpu030` repo `ed0761d7a349fb34b201071f98ac88b6d91cafe2` driver `580.126.09` cuda `13.0`
  - job `56693863` host `gpu030` repo `ed0761d7a349fb34b201071f98ac88b6d91cafe2` driver `580.126.09` cuda `13.0`
  - job `56693864` host `gpu030` repo `ed0761d7a349fb34b201071f98ac88b6d91cafe2` driver `580.126.09` cuda `13.0`
  - job `56693865` host `gpu030` repo `ed0761d7a349fb34b201071f98ac88b6d91cafe2` driver `580.126.09` cuda `13.0`
  - job `56693866` host `gpu030` repo `ed0761d7a349fb34b201071f98ac88b6d91cafe2` driver `580.126.09` cuda `13.0`
  - error `/work/pi_mengfanxu_umass_edu/bhuo_umass_edu/infom-learning-from-obs/runs/unity_cube_single_paper_repro_retry_exclude_gpu026_20260504_2236/slurm/infom-repro-t2-s0-56693859.err:2`: Traceback (most recent call last):
  - error `/work/pi_mengfanxu_umass_edu/bhuo_umass_edu/infom-learning-from-obs/runs/unity_cube_single_paper_repro_retry_exclude_gpu026_20260504_2236/slurm/infom-repro-t2-s0-56693859.err:11`: jaxlib._jax.XlaRuntimeError: INTERNAL: no supported devices found for platform CUDA
  - error `/work/pi_mengfanxu_umass_edu/bhuo_umass_edu/infom-learning-from-obs/runs/unity_cube_single_paper_repro_retry_exclude_gpu026_20260504_2236/slurm/infom-repro-t2-s0-56693859.err:15`: jax.errors.SimplifiedTraceback: For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set JAX_TRACEBACK_FILTERING=off to include these.
  - error `/work/pi_mengfanxu_umass_edu/bhuo_umass_edu/infom-learning-from-obs/runs/unity_cube_single_paper_repro_retry_exclude_gpu026_20260504_2236/slurm/infom-repro-t2-s0-56693859.err:19`: Traceback (most recent call last):
  - error `/work/pi_mengfanxu_umass_edu/bhuo_umass_edu/infom-learning-from-obs/runs/unity_cube_single_paper_repro_retry_exclude_gpu026_20260504_2236/slurm/infom-repro-t2-s0-56693859.err:48`: RuntimeError: Unable to initialize backend 'cuda': INTERNAL: no supported devices found for platform CUDA (you may need to uninstall the failing plugin package, or set JAX_PLATFORMS=cpu to skip this backend.)
  - error `/work/pi_mengfanxu_umass_edu/bhuo_umass_edu/infom-learning-from-obs/runs/unity_cube_single_paper_repro_retry_exclude_gpu026_20260504_2236/slurm/infom-repro-t2-s1-56693860.err:2`: Traceback (most recent call last):
  - error `/work/pi_mengfanxu_umass_edu/bhuo_umass_edu/infom-learning-from-obs/runs/unity_cube_single_paper_repro_retry_exclude_gpu026_20260504_2236/slurm/infom-repro-t2-s1-56693860.err:11`: jaxlib._jax.XlaRuntimeError: INTERNAL: no supported devices found for platform CUDA
  - error `/work/pi_mengfanxu_umass_edu/bhuo_umass_edu/infom-learning-from-obs/runs/unity_cube_single_paper_repro_retry_exclude_gpu026_20260504_2236/slurm/infom-repro-t2-s1-56693860.err:15`: jax.errors.SimplifiedTraceback: For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set JAX_TRACEBACK_FILTERING=off to include these.
  - error `/work/pi_mengfanxu_umass_edu/bhuo_umass_edu/infom-learning-from-obs/runs/unity_cube_single_paper_repro_retry_exclude_gpu026_20260504_2236/slurm/infom-repro-t2-s1-56693860.err:19`: Traceback (most recent call last):
  - error `/work/pi_mengfanxu_umass_edu/bhuo_umass_edu/infom-learning-from-obs/runs/unity_cube_single_paper_repro_retry_exclude_gpu026_20260504_2236/slurm/infom-repro-t2-s1-56693860.err:48`: RuntimeError: Unable to initialize backend 'cuda': INTERNAL: no supported devices found for platform CUDA (you may need to uninstall the failing plugin package, or set JAX_PLATFORMS=cpu to skip this backend.)

## Notes

- Paper metric here is mean success at 1.4M, 1.45M, and 1.5M total steps.
- Best success is diagnostic only and is not the paper metric.
- Validation spikes are flagged as diagnostics; they are not by themselves proof of failed policy learning.
