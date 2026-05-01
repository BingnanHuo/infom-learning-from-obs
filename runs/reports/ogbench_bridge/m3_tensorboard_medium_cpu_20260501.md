# M3 OGBench Bridge Cube-Single Local Matched Runs - 2026-05-01

## Goal

Run a local matched cube-single comparison between ego-only InFOM and bridge-aware InFOM while Unity is unavailable. Use matched seeds and matched bridge datasets. Add TensorBoard tracking so training can be monitored live.

## Code State

- Branch: `feature/tensorboard-local-ogbench-runs`
- TensorBoard logging added behind `--enable_tensorboard=1`.
- Default event directory: `<save_dir>/<wandb_run_group>/<run_name>/tensorboard`.
- TensorBoard dependencies added to `requirements.txt`.
- TensorBoard CLI requires `pkg_resources` with the currently installed TensorBoard, so `setuptools` is pinned to `<81`.

## Resource Gate

- Initial GPU check showed active compute jobs, so the first 20k/10k matched run used CPU.
- GPU became briefly free during data planning, but later showed active `mmmarl-mp311` compute processes again before medium training.
- Medium matched training therefore used CPU-only execution:
  - `CUDA_VISIBLE_DEVICES=`
  - `JAX_PLATFORMS=cpu`
  - `JAX_PLATFORM_NAME=cpu`
- CPU and memory were acceptable during runs. Medium bridge-aware CPU runs were the expensive training condition.

## TensorBoard

Medium run command:

```bash
conda run -n infom-obs tensorboard --logdir exp/m3_bridge_tensorboard_medium/20260501-161629/runs --host 127.0.0.1 --port 6006
```

Tiny run command:

```bash
conda run -n infom-obs tensorboard --logdir exp/m3_bridge_tensorboard_long/20260501-160116/runs --host 127.0.0.1 --port 6006
```

Main tags to watch:

- `evaluation/episode.return`: good if it rises above `-200`; flat `-200` means no task progress.
- `evaluation/episode.success` and `evaluation/success`: any sustained nonzero value is meaningful.
- `evaluation/episode.length`: only useful together with success; shorter failed episodes are not good performance.
- `training/bridge/weighted_bridge_loss` and `validation/bridge/weighted_bridge_loss`: bridge health checks only. They should stay finite and ideally decrease or stabilize.
- `training/grad/norm`, `training/bc/mse`, `validation/bc/mse`, `training/flow_occupancy/neg_elbo_loss`: instability and overfit checks.

TensorBoard inspection verified scalar event files for all medium runs, including eval tags and bridge-only tags for bridge-aware runs.

## Data Artifacts

Tiny pilot data:

- Path: `exp/m3_bridge_cube_single_pilot/20260501-135935/data`
- Pretraining train/val: 4,000 / 400 transitions
- Finetuning train/val: 2,000 / 200 transitions

Medium local data:

- Path: `exp/m3_bridge_cube_single_medium/20260501-161629/data`
- Pretraining train: `bridge-cube-single-play-v0.npz`, 100,100 transitions
- Pretraining val: `bridge-cube-single-play-v0-val.npz`, 10,010 transitions
- Finetuning train: `bridge-cube-single-play-ft-v0.npz`, 20,020 transitions
- Finetuning val: `bridge-cube-single-play-ft-v0-val.npz`, 2,002 transitions
- Pretraining generation wall time: 2:10:27 for 110 rendered episodes
- Finetuning generation wall time: 26:03 for 22 rendered episodes

Data generation is the current local bottleneck: about 70-74 seconds per rendered episode with the current local settings.

## Run Configuration

Both run sets used a reduced local CPU profile:

- Env: `bridge-cube-single-play-singletask-task1-v0`
- Seeds: `0`, `1`
- Eval episodes: `5`
- Batch size: `32`
- Hidden dims: `(64,64)` for actor, value, reward, and intention encoders
- Latent dim: `32`
- Flow goals: `4`
- Flow steps: `4`
- Actor frequency: `1`
- Bridge-aware only: `--agent.bridge_loss_weight=0.1 --agent.bridge_encoder=impala_debug`

Tiny matched run:

- Root: `exp/m3_bridge_tensorboard_long/20260501-160116/runs`
- Steps: 20,000 pretraining + 10,000 finetuning
- Data: tiny pilot data

Medium matched run:

- Root: `exp/m3_bridge_tensorboard_medium/20260501-161629/runs`
- Steps: 100,000 pretraining + 50,000 finetuning
- Data: medium local data

## Results

### Tiny 20k/10k Run

| Variant | Seeds | Final return mean +/- sd | Final success mean +/- sd | Best return mean +/- sd | Best success mean +/- sd |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ego-only InFOM | 2 | -200.0 +/- 0.0 | 0.0 +/- 0.0 | -200.0 +/- 0.0 | 0.0 +/- 0.0 |
| Bridge-aware InFOM | 2 | -200.0 +/- 0.0 | 0.0 +/- 0.0 | -200.0 +/- 0.0 | 0.0 +/- 0.0 |

Bridge health:

- Seed 0 weighted bridge loss: `0.0076867924 -> 0.00017277052`
- Seed 1 weighted bridge loss: `0.00796297 -> 0.00011627783`

### Medium 100k/50k Run

| Variant | Seeds | Final return mean +/- sd | Final success mean +/- sd | Best return mean +/- sd | Best success mean +/- sd |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ego-only InFOM | 2 | -200.0 +/- 0.0 | 0.0 +/- 0.0 | -200.0 +/- 0.0 | 0.0 +/- 0.0 |
| Bridge-aware InFOM | 2 | -200.0 +/- 0.0 | 0.0 +/- 0.0 | -200.0 +/- 0.0 | 0.0 +/- 0.0 |

Bridge health:

- Seed 0 weighted bridge loss: `0.0014154118 -> 0.00001240642`
- Seed 0 validation weighted bridge loss final: `0.00006917616`
- Seed 1 weighted bridge loss: `0.0020049484 -> 0.000014067596`
- Seed 1 validation weighted bridge loss final: `0.00012504698`

## Interpretation

This is negative/inconclusive task-performance evidence. Neither ego-only nor bridge-aware InFOM achieved nonzero success, and both remained at the minimum observed eval return of `-200.0` across matched seeds and both local run scales.

The bridge auxiliary objective is technically training: bridge losses are finite and decrease by roughly two orders of magnitude. That does not translate into task success in these local runs. The correct claim is that the bridge-aware training path, TensorBoard logging, data routing, and matched-run pipeline work locally, but there is not yet evidence that bridge-aware InFOM improves cube-single performance.

Likely limitations:

- The local profile is intentionally reduced from repo defaults and README-scale examples.
- The medium finetune dataset is 20,020 transitions, not the 500-episode README-scale finetune set.
- Only two seeds were run.
- CPU-only bridge training used `impala_debug`, not a full bridge encoder profile.
- Full default training scale is much larger than these local runs.

## Next Steps

1. Do not claim a performance win from M3 yet.
2. Reuse `exp/m3_bridge_cube_single_medium/20260501-161629/data` for follow-up runs instead of regenerating it.
3. When GPU is free, run a single-seed fuller-profile bridge vs ego comparison with README InFOM hyperparameters:
   - `--agent.expectile=0.95`
   - `--agent.kl_weight=0.05`
   - `--agent.alpha=30`
   - larger/default model settings if runtime allows
4. Improve data-generation ergonomics before scaling:
   - add periodic progress logging that survives redirection
   - test whether EGL rendering is faster than the current local path
   - consider chunked/incremental output so long generation is recoverable
5. Once Unity is available again, move full 250k/100k or default-scale matched seeds to the cluster.
