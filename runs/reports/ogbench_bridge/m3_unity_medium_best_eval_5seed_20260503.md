# M3 Unity OGBench Bridge Medium Best-Eval 5-Seed - 2026-05-03

## Goal

Run the follow-up Unity matched cube-single comparison between ego-only InFOM and bridge-aware InFOM with best-eval checkpoint tracking enabled and five matched seeds.

## Artifact Roots

- Remote repo: `/work/pi_mengfanxu_umass_edu/bhuo_umass_edu/repos/infom-learning-from-obs`
- Remote branch: `feature/tensorboard-local-ogbench-runs`
- Remote commit: `04cbbf6`
- Run root: `/work/pi_mengfanxu_umass_edu/bhuo_umass_edu/infom-learning-from-obs/runs/unity_m3_medium_best_eval_5seed_20260503`
- Data root: `/work/pi_mengfanxu_umass_edu/bhuo_umass_edu/infom-learning-from-obs/data/m3_bridge_cube_single_medium_20260501-161629`
- Summary command: `python scripts/summarize_ogbench_bridge_runs.py /work/pi_mengfanxu_umass_edu/bhuo_umass_edu/infom-learning-from-obs/runs/unity_m3_medium_best_eval_5seed_20260503 --slurm-root /work/pi_mengfanxu_umass_edu/bhuo_umass_edu/infom-learning-from-obs/runs/unity_m3_medium_best_eval_5seed_20260503/slurm`

## Run Configuration

- Environment: `bridge-cube-single-play-singletask-task1-v0`
- Variants: ego-only InFOM with `bridge_loss_weight=0.0`; bridge-aware InFOM with `bridge_loss_weight=0.1`
- Seeds: `0`, `1`, `2`, `3`, `4` for both variants
- Training scale: 250k pretraining steps plus 200k finetuning steps
- Eval cadence: every 10k finetuning steps, 20 eval episodes
- InFOM hyperparameters: `expectile=0.95`, `kl_weight=0.05`, `alpha=30`
- Bridge encoder: `impala_small`
- TensorBoard: enabled
- W&B: disabled
- Best-eval checkpointing: enabled with `best_eval_metric=evaluation/episode.return` and `best_eval_mode=max`

## Completion And Health

Submitted jobs:

| Variant | Seed | Job ID | Slurm state | Exit code | Elapsed |
| --- | ---: | ---: | --- | ---: | ---: |
| Ego-only | 0 | 56631205 | COMPLETED | 0:0 | 00:26:31 |
| Bridge-aware | 0 | 56631206 | COMPLETED | 0:0 | 01:07:42 |
| Ego-only | 1 | 56631207 | COMPLETED | 0:0 | 00:26:58 |
| Bridge-aware | 1 | 56631208 | COMPLETED | 0:0 | 00:38:34 |
| Ego-only | 2 | 56631209 | COMPLETED | 0:0 | 00:25:50 |
| Bridge-aware | 2 | 56631210 | COMPLETED | 0:0 | 00:38:32 |
| Ego-only | 3 | 56631211 | COMPLETED | 0:0 | 00:25:02 |
| Bridge-aware | 3 | 56631212 | COMPLETED | 0:0 | 00:39:03 |
| Ego-only | 4 | 56631213 | COMPLETED | 0:0 | 00:26:54 |
| Bridge-aware | 4 | 56631214 | COMPLETED | 0:0 | 00:46:25 |

The summarizer parsed 10 run directories and found zero fatal Slurm error-pattern hits. Raw stderr logs contain repeated MuJoCo/GLFW messages about missing `DISPLAY`; these are nonfatal headless-rendering warnings and all jobs completed with exit code `0:0`.

All runs produced `finetuning_eval.csv`, `pretraining_train.csv`, `flags.json`, `best_eval.json`, and at least two checkpoints. Every run has 21 evaluation rows through final step `450000`.

## Results

Aggregate matched-seed metrics:

| Variant | Seeds | Final return mean +/- sd | Final success mean +/- sd | Best return mean +/- sd | Best success mean +/- sd |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bridge-aware InFOM | 5 | -191.55 +/- 5.08 | 0.100 +/- 0.061 | -184.93 +/- 4.47 | 0.200 +/- 0.050 |
| Ego-only InFOM | 5 | -192.42 +/- 9.81 | 0.090 +/- 0.074 | -182.56 +/- 6.17 | 0.200 +/- 0.061 |

Matched bridge-minus-ego deltas:

| Seed | Final return delta | Final success delta | Best return delta | Best success delta |
| ---: | ---: | ---: | ---: | ---: |
| 0 | -1.45 | 0.000 | -7.15 | -0.050 |
| 1 | 6.10 | 0.000 | 3.60 | 0.050 |
| 2 | 10.55 | 0.200 | -4.60 | 0.050 |
| 3 | -10.15 | -0.100 | -3.90 | -0.050 |
| 4 | -0.70 | -0.050 | 0.20 | 0.000 |

Mean bridge-minus-ego deltas: final return `0.87 +/- 7.91`, final success `0.010 +/- 0.114`, best return `-2.37 +/- 4.25`, best success `0.000 +/- 0.050`.

Per-seed evaluation summary:

| Variant | Seed | Job ID | Eval rows | Final step | Final return | Final success | Best return | Best success | Checkpoints | Best eval step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bridge-aware | 0 | 56631206 | 21 | 450000 | -198.10 | 0.050 | -185.75 | 0.150 | 4 | 300000 |
| Bridge-aware | 1 | 56631208 | 21 | 450000 | -189.50 | 0.100 | -184.35 | 0.250 | 6 | 370000 |
| Bridge-aware | 2 | 56631210 | 21 | 450000 | -189.45 | 0.200 | -189.45 | 0.200 | 7 | 450000 |
| Bridge-aware | 3 | 56631212 | 21 | 450000 | -185.40 | 0.100 | -177.70 | 0.250 | 5 | 390000 |
| Bridge-aware | 4 | 56631214 | 21 | 450000 | -195.30 | 0.050 | -187.40 | 0.150 | 4 | 440000 |
| Ego-only | 0 | 56631205 | 21 | 450000 | -196.65 | 0.050 | -178.60 | 0.200 | 5 | 430000 |
| Ego-only | 1 | 56631207 | 21 | 450000 | -195.60 | 0.100 | -187.95 | 0.200 | 5 | 340000 |
| Ego-only | 2 | 56631209 | 21 | 450000 | -200.00 | 0.000 | -184.85 | 0.150 | 2 | 410000 |
| Ego-only | 3 | 56631211 | 21 | 450000 | -175.25 | 0.200 | -173.80 | 0.300 | 4 | 390000 |
| Ego-only | 4 | 56631213 | 21 | 450000 | -194.60 | 0.100 | -187.60 | 0.150 | 4 | 380000 |

Bridge objective health:

| Seed | Training weighted bridge loss first | Training weighted bridge loss last | Validation weighted bridge loss last |
| ---: | ---: | ---: | ---: |
| 0 | 0.0014678274 | 0.0000574434 | 0.0024332271 |
| 1 | 0.0014784373 | 0.0000506162 | 0.0027503580 |
| 2 | 0.0013027111 | 0.0000607988 | 0.0026365134 |
| 3 | 0.0019727808 | 0.0000373645 | 0.0013354834 |
| 4 | 0.0014935763 | 0.0000352225 | 0.0017073680 |

## Interpretation

This is mixed evidence, not a reliable bridge-aware downstream-control win.

Final evaluation slightly favors bridge-aware InFOM on mean return and success: final return `-191.55` vs ego-only `-192.42`, and final success `0.100` vs ego-only `0.090`. The matched bridge-minus-ego final deltas are small relative to seed variance: `0.87 +/- 7.91` return and `0.010 +/- 0.114` success.

Best-over-training return favors ego-only: bridge-aware `-184.93 +/- 4.47` vs ego-only `-182.56 +/- 6.17`. Best success is tied at `0.200` mean, with similar uncertainty. Since both variants still show sparse, noisy success, final-only claims remain fragile even with five matched seeds.

The bridge auxiliary objective is active and trains down for every bridge-aware seed. That supports the implementation path and data plumbing, but it still does not prove a robust downstream advantage on this medium cube-single setting.

## Next Steps

1. Do not claim a bridge-aware performance win from this campaign.
2. Treat this as pipeline evidence plus inconclusive performance evidence.
3. If more time is available, run a larger or longer-horizon matched campaign with the same best-eval reporting and keep final metrics separate from best-over-training metrics.
4. Investigate why both variants remain close to the sparse-success floor before scaling beyond this task.
