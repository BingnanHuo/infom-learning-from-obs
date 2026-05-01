# M3 Cube-Single Bridge Pilot - 2026-05-01

## Summary

This run is a local CPU pilot for the first matched OGBench bridge comparison. It verifies that the merged bridge-aware InFOM path can train and evaluate against generated paired cube-single bridge data.

This is pipeline evidence, not a final performance result. The GPU was occupied by other compute jobs, so all training was forced to CPU.

## Resource State

- Unity cluster: unavailable.
- GPU: RTX 4090 was busy at about 96-97% utilization with active external compute jobs.
- CPU: 32 cores, load average about 4-10 during setup and pilot.
- RAM: about 12-14 GiB available.
- Execution mode: CPU forced with `CUDA_VISIBLE_DEVICES=` and `JAX_PLATFORMS=cpu`.

JAX still printed a CUDA plugin discovery warning when CUDA was hidden, then continued on CPU. Runs completed successfully.

## Data

Generated locally under ignored output:

`exp/m3_bridge_cube_single_pilot/20260501-135935/data`

Files:

| File | Transitions | Third-person shape |
| --- | ---: | --- |
| `bridge-cube-single-play-v0.npz` | 4000 | `(64, 64, 3)` |
| `bridge-cube-single-play-v0-val.npz` | 400 | `(64, 64, 3)` |
| `bridge-cube-single-play-ft-v0.npz` | 2000 | `(64, 64, 3)` |
| `bridge-cube-single-play-ft-v0-val.npz` | 200 | `(64, 64, 3)` |

Loader smoke passed for `bridge-cube-single-play-singletask-task1-v0` with `reward_free=True`.

## Runs

Common settings:

- Env: `bridge-cube-single-play-singletask-task1-v0`
- Seeds: `0, 1`
- Pretraining steps: `500`
- Fine-tuning steps: `250`
- Eval episodes: `2`
- Hidden dims: `(64, 64)`
- Latent dim: `32`
- Flow goals: `4`
- Flow steps: `4`
- Bridge encoder for bridge variant: `impala_debug`

| Variant | Seed | Run directory | Final eval step | Episode return | Episode success | Bridge metrics |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| baseline | 0 | `exp/m3_bridge_cube_single_pilot/20260501-135935/runs/pilot_cpu_baseline/sd000_20260501_140922` | 750 | -200.0 | 0.0 | absent |
| baseline | 1 | `exp/m3_bridge_cube_single_pilot/20260501-135935/runs/pilot_cpu_baseline/sd001_20260501_140940` | 750 | -200.0 | 0.0 | absent |
| bridge | 0 | `exp/m3_bridge_cube_single_pilot/20260501-135935/runs/pilot_cpu_bridge/sd000_20260501_140957` | 750 | -200.0 | 0.0 | pretraining only |
| bridge | 1 | `exp/m3_bridge_cube_single_pilot/20260501-135935/runs/pilot_cpu_bridge/sd001_20260501_141028` | 750 | -200.0 | 0.0 | pretraining only |

Bridge pretraining metrics were logged only for the bridge-aware variant:

| Seed | `training/bridge/bridge_consistency_loss` | `training/bridge/weighted_bridge_loss` |
| ---: | ---: | ---: |
| 0 | 0.19683759 | 0.01968376 |
| 1 | 0.18213542 | 0.018213542 |

## Interpretation

- The generated paired bridge data loads correctly through the bridge-aware OGBench path.
- The matched baseline and bridge-aware variants complete identical CPU pilot schedules.
- The baseline has no bridge metrics, and the bridge-aware variant logs bridge metrics during pretraining only.
- The 500/250-step CPU pilot is too short to produce task success; both variants have final episode return `-200.0` and success `0.0`.

## Next Step

When the local GPU is free, rerun the same matched comparison with a larger GPU profile, keeping the generated dataset fixed:

- Pretraining steps: at least `5000`
- Fine-tuning steps: at least `2000`
- Eval episodes: at least `5`
- Seeds: `0, 1`, then expand if the curves are stable
