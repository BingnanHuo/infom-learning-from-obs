# Phase 1: Self Third-Person Bridge

## Goal

Extend the InFOM setup toward the class-project question: can a latent intent representation learned from third-person observations improve first-person control when observation and control spaces do not align cleanly?

The first implementation phase uses the acting agent's own synchronized third-person view as a bridge. This is the shortest path to a meaningful cross-view intent experiment while still differing materially from upstream InFOM.

## First Benchmark Choice

Start with the state-control OGBench cube task:

- offline pretraining data: `cube-single-play-v0`
- future online task env: `cube-single-singletask-task1-v0`
- auxiliary bridge view: `front_pixels` rendered at `64x64`

Reasons:

- it treats the upstream InFOM control observation as the ego view, matching the intended project interpretation
- the state env can already render the same scene through the `front_pixels` camera without adding a new simulator stack
- it preserves the option to move from offline pretraining into online bridge fine-tuning without redefining the actor-facing observation

Isaac Lab remains a later extension, not the bootstrap target.

## Phase-1 Data Contract

The paired sample schema for the bridge stage is:

- `ego_obs`
- `third_person_obs`
- `ego_action`
- `ego_next_obs`
- `qpos`
- `qvel`
- `episode_id`
- `timestep`

For phase 1, the dataset family is implicitly `self_bridge`; no separate `view_source` field is required yet.

## Intended Training Shape

Phase 1 should preserve the upstream InFOM structure as much as possible:

1. keep the upstream intent model and encoder stack as the base
2. add a paired-view data path that exposes state-control ego observations and synchronized third-person renders from the same behavior
3. use the paired offline dataset for bridge pretraining
4. extend later into online fine-tuning where the acting agent receives the standard ego observation while third-person renders remain available as auxiliary bridge information

## Baseline Order

The first comparison set should stay small:

- upstream-like first-person baseline with no cross-view latent
- observational encoder baseline without a reusable intent-conditioned control variable
- shared-latent bridge model

## Not In Phase 1

Do not include these in the first implementation slice:

- removal of the self third-person bridge
- cross-embodiment transfer
- Isaac Lab-specific simulator work
- large benchmark sweeps

## Practical Constraint

The default downloaded OGBench state dataset is not sufficient for offline paired-view replay because it does not retain `qpos` and `qvel`.

That means the first code slice must custom-collect a bridge dataset rather than trying to retrofit the stock `cube-single-play-v0` download into synchronized third-person pixels.
