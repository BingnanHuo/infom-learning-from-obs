# M3 Unity OGBench Bridge Medium Full Profile - 2026-05-02

## Goal

Run the first real Unity matched cube-single comparison between ego-only InFOM and bridge-aware InFOM with matched seeds and matched bridge datasets.

## Artifact Roots

- Remote repo: `/work/pi_mengfanxu_umass_edu/bhuo_umass_edu/repos/infom-learning-from-obs`
- Remote branch: `feature/tensorboard-local-ogbench-runs`
- Remote commit: `deb4d99`
- Run root: `/work/pi_mengfanxu_umass_edu/bhuo_umass_edu/infom-learning-from-obs/runs/unity_m3_medium_fullprofile_20260502`
- Data root: `/work/pi_mengfanxu_umass_edu/bhuo_umass_edu/infom-learning-from-obs/data/m3_bridge_cube_single_medium_20260501-161629`
- TensorBoard event files: one `tensorboard/events.out.tfevents*` file per run directory under the run root.

## Run Configuration

- Environment: `bridge-cube-single-play-singletask-task1-v0`
- Variants: ego-only InFOM with `bridge_loss_weight=0.0`; bridge-aware InFOM with `bridge_loss_weight=0.1`
- Seeds: `0`, `1`, `2` for both variants
- Training scale: 250k pretraining steps plus 100k finetuning steps
- Eval cadence: every 10k finetuning steps, 20 eval episodes
- InFOM hyperparameters: `expectile=0.95`, `kl_weight=0.05`, `alpha=30`
- Bridge encoder: `impala_small`
- TensorBoard: enabled
- W&B: disabled
- Checkpoint cadence: `save_interval=999999`, so these runs do not preserve intermediate checkpoints for best-checkpoint selection.

Dataset counts:

| Split | File | Transitions |
| --- | --- | ---: |
| Pretrain train | `bridge-cube-single-play-v0.npz` | 100,100 |
| Pretrain val | `bridge-cube-single-play-v0-val.npz` | 10,010 |
| Finetune train | `bridge-cube-single-play-ft-v0.npz` | 20,020 |
| Finetune val | `bridge-cube-single-play-ft-v0-val.npz` | 2,002 |

## Completion And Health

Submitted jobs:

| Variant | Seed | Job ID | Completion evidence |
| --- | ---: | ---: | --- |
| Ego-only | 0 | 56533572 | Slurm log reached `350000/350000` |
| Ego-only | 1 | 56533573 | Slurm log reached `350000/350000` |
| Ego-only | 2 | 56533574 | Slurm log reached `350000/350000` |
| Bridge-aware | 0 | 56533575 | Slurm log reached `350000/350000` |
| Bridge-aware | 1 | 56533578 | Slurm log reached `350000/350000` |
| Bridge-aware | 2 | 56533579 | Slurm log reached `350000/350000` |

`sacct` returned no rows for these job IDs at inspection time, so completion is based on run artifacts, Slurm log tails, and the fact that the jobs are absent from `squeue`. No `Traceback`, `RuntimeError`, OOM, timeout, cancellation, or failure patterns were found in the run's Slurm logs.

TensorBoard scalar checks:

- All six runs include `evaluation/episode.return` and `evaluation/episode.success`, with 11 scalar points from step `250001` through `350000`.
- Bridge-aware runs additionally include `training/bridge/weighted_bridge_loss`, with 50 scalar points from step `5000` through `250000`.

## Results

Aggregate matched-seed metrics:

| Variant | Seeds | Final return mean +/- sd | Final success mean +/- sd | Best return mean +/- sd | Best success mean +/- sd |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ego-only InFOM | 3 | -200.00 +/- 0.00 | 0.000 +/- 0.000 | -187.63 +/- 0.71 | 0.133 +/- 0.029 |
| Bridge-aware InFOM | 3 | -192.35 +/- 10.18 | 0.100 +/- 0.100 | -189.57 +/- 7.66 | 0.117 +/- 0.076 |

Per-seed evaluation summary:

| Variant | Seed | Final return | Final success | Best return | Best success | Run time evidence |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Ego-only | 0 | -200.00 | 0.00 | -188.45 | 0.15 | `350000/350000 [37:02]` |
| Ego-only | 1 | -200.00 | 0.00 | -187.25 | 0.15 | `350000/350000 [36:05]` |
| Ego-only | 2 | -200.00 | 0.00 | -187.20 | 0.10 | `350000/350000 [37:18]` |
| Bridge-aware | 0 | -196.25 | 0.10 | -194.95 | 0.10 | `350000/350000 [27:25]` |
| Bridge-aware | 1 | -200.00 | 0.00 | -192.95 | 0.05 | `350000/350000 [52:40]` |
| Bridge-aware | 2 | -180.80 | 0.20 | -180.80 | 0.20 | `350000/350000 [53:28]` |

Bridge objective health:

| Seed | Training weighted bridge loss first | Training weighted bridge loss last | Validation weighted bridge loss last |
| ---: | ---: | ---: | ---: |
| 0 | 0.0014645666 | 0.0000436401 | 0.0017868329 |
| 1 | 0.0014485967 | 0.0001216440 | 0.0054601110 |
| 2 | 0.0012972661 | 0.0000313688 | 0.0012876412 |

## Interpretation

This is mixed preliminary evidence, not a clean bridge-aware win.

The bridge-aware variant has better final mean return and final mean success than ego-only on this run: final return `-192.35` vs `-200.00`, and final success `0.100` vs `0.000`. That improvement is driven mostly by seed 2; bridge seed 1 still ends at zero success.

The ego-only variant has slightly better best-over-training mean metrics: best return `-187.63` vs bridge-aware `-189.57`, and best success `0.133` vs bridge-aware `0.117`. This matters because both variants show unstable sparse successes during finetuning, and final-eval-only conclusions are fragile at three seeds.

The bridge auxiliary objective is training: weighted bridge loss decreases for all bridge-aware seeds. That establishes that the bridge stream is being consumed and optimized, but it does not yet establish a reliable downstream control advantage.

## Next Steps

1. Do not claim a bridge-aware performance win from this run alone.
2. Add checkpointing or best-eval checkpoint selection before the next serious comparison; the current `save_interval=999999` prevents auditing best checkpoints.
3. Run a larger confirmation campaign with at least five matched seeds, keeping final and best-over-training metrics separate.
4. Consider a README-scale dataset or longer finetuning horizon, because both variants still spend many evals near the `-200` floor.
5. Keep TensorBoard active and watch `evaluation/episode.return`, `evaluation/episode.success`, and bridge-only `training/bridge/weighted_bridge_loss` for bridge runs.
