# Phase 1: Self Third-Person Bridge

## Goal

Extend the InFOM setup toward the class-project question: can a latent intent representation learned from third-person observations improve first-person control when observation and control spaces do not align cleanly?

The first implementation phase uses the acting agent's own synchronized third-person view as a bridge. This is the shortest path to a meaningful cross-view intent experiment while still differing materially from upstream InFOM.

## First Benchmark Choice

Start with `visual-cube-single-play-singletask-task1-v0` from OGBench.

Reasons:

- it keeps the upstream JAX/Flax code and image encoder path relevant
- it avoids the setup cost of Isaac Lab during the highest-risk week of the class schedule
- it is sufficient for validating the paired-view data path and the shared-latent training objective

Isaac Lab remains a later extension, not the bootstrap target.

## Phase-1 Data Contract

The paired sample schema for the bridge stage is:

- `ego_obs`
- `ego_action`
- `ego_next_obs`
- `third_person_obs`
- `episode_id`
- `timestep`
- `view_source`

For phase 1, `view_source` is restricted to `self_bridge`.

## Intended Training Shape

Phase 1 should preserve the upstream InFOM structure as much as possible:

1. keep the upstream intent model and encoder stack as the base
2. add a paired-view data path that exposes first-person and synchronized third-person observations from the same behavior
3. learn a shared latent space that is encouraged to align strategy across views rather than raw pixels alone
4. condition downstream first-person control on the shared latent

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
