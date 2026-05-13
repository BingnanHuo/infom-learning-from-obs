# Final Project Wrap-Up Audit - 2026-05-12

## Scope

This is the final compact audit for the current `infom-learning-from-obs`
project package before preparing the merge to `main`.

Inspected state:

- Local repo: `/home/nick/infom-learning-from-obs`
- Final merge worktree: `/tmp/infom-final-project-wrapup`
- Origin repo: `BingnanHuo/infom-learning-from-obs`
- Merged baseline on `main`: `e9f532b715ac43219a271944b4cde1d9c93fd9c0`
- Prior feature tip: `31ecf2f9a14fb925266774b24319bf0305e1092a`
- Unity repo: `/work/pi_mengfanxu_umass_edu/bhuo_umass_edu/repos/infom-learning-from-obs`
- Durable Unity run root: `/work/pi_mengfanxu_umass_edu/bhuo_umass_edu/infom-learning-from-obs/runs`

## Unity Cluster State

Final scan time: `2026-05-12T23:43:40+00:00`.

- `squeue` showed no active jobs matching `infom`, `ogbench`, `bridge`,
  `state`, `tcn`, or `checkpoint`.
- `squeue` did show unrelated held `ada-hp-*` jobs in `JobHeldUser`; these
  were not part of this project.
- The Unity repo was still on `feature/tensorboard-local-ogbench-runs` at
  `31ecf2f9a14fb925266774b24319bf0305e1092a` with dirty final-project files.
- The Unity clone's `origin/main` ref was stale during the audit
  (`b82b32b8b7188c01ab6154794978ef517a00b37d`), while local `origin/main`
  had already been fetched to `e9f532b715ac43219a271944b4cde1d9c93fd9c0`.
  Update Unity from Git after the final package PR lands.

## Synced Compact Artifacts

Synced from Unity durable reports to local:

- `runs/reports/ogbench_paper_repro/diagnosis_ab_20260505_2026.md`
- `runs/reports/ogbench_paper_repro/diagnosis_ab_20260505_2026.records.json`

Already present and included locally:

- `runs/reports/ogbench_bridge/*.md`
- `runs/reports/ogbench_paper_repro/*.md`
- `runs/reports/ogbench_paper_repro/*.json`
- `runs/unity_state_distilled_method_a/**/manifest.json`
- `runs/unity_state_distilled_method_a_perf/**/manifest.json`
- `runs/unity_cross_modal_tcn_method_b/**/manifest.json`
- `runs/unity_checkpoint_demos/**/manifest.json`
- `output/report/figures/*.png`
- Executed/source notebooks under `output/jupyter-notebook/*.ipynb`
- `docs/final-report.tex`, `docs/final-report.pdf`, and bibliography/style files

Intentionally excluded from the merge package:

- Raw checkpoints
- TensorBoard event files
- Notebook caches and raw artifact/log/video directories
- `runs/tmp`
- LaTeX build intermediates
- Python cache files
- External/source PDFs that are not required to rebuild the final report

## Result Evidence

### M3 Bridge-Aware InFOM

Primary report:
`runs/reports/ogbench_bridge/m3_unity_medium_best_eval_5seed_20260503.md`

The five matched-seed Unity run completed for both ego-only and bridge-aware
variants. All ten jobs completed with Slurm exit code `0:0`, each run produced
evaluation CSVs, flags, `best_eval.json`, and checkpoints, and the summarizer
found no fatal Slurm error-pattern hits.

Headline matched-seed metrics:

| Variant | Seeds | Final return mean +/- sd | Final success mean +/- sd | Best return mean +/- sd | Best success mean +/- sd |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bridge-aware InFOM | 5 | -191.55 +/- 5.08 | 0.100 +/- 0.061 | -184.93 +/- 4.47 | 0.200 +/- 0.050 |
| Ego-only InFOM | 5 | -192.42 +/- 9.81 | 0.090 +/- 0.074 | -182.56 +/- 6.17 | 0.200 +/- 0.061 |

Mean bridge-minus-ego deltas were small relative to variance: final return
`0.87 +/- 7.91`, final success `0.010 +/- 0.114`, best return
`-2.37 +/- 4.25`, and best success `0.000 +/- 0.050`.

Interpretation: this is pipeline evidence and mixed performance evidence, not a
reliable bridge-aware control win.

### OGBench Paper-Reproduction Gate

Primary reports:

- `runs/reports/ogbench_paper_repro/diagnosis_ab_20260505_2026.md`
- `runs/reports/ogbench_paper_repro/diagnosis_20260505_combined.md`
- `runs/reports/ogbench_paper_repro/diagnosis_20260505_unity_existing.md`

The A/B gate reproduced some high best-checkpoint success, but paper-window
success was far below the paper target in the checked A100 task-1 comparison.
The report records paper-window success `16.00 +/- 7.54` versus target
`92.5 +/- 4.0`, with final success `32.00 +/- 8.49`.

The run health table flags late collapse and large validation/gradient
diagnostics. Best success is useful for diagnosis but is not the paper metric.

### Method A: State-Distilled Cross-Modal InFOM

Evidence package:

- `agents/cross_modal_state_distilled_infom.py`
- `tests/test_state_distilled_infom.py`
- `output/jupyter-notebook/state-distilled-method-a-monitor.executed.ipynb`
- `output/jupyter-notebook/method-b-ft-collapse-analysis.executed.ipynb`
- `output/jupyter-notebook/final-project-presentation.executed.ipynb`
- `runs/unity_state_distilled_method_a/**/manifest.json`
- `runs/unity_state_distilled_method_a_perf/**/manifest.json`

The evidence supports that the state-distilled bridge can learn a state-like
latent and produce real downstream control signal. The executed final
presentation notebook records task 1 seed 5 with paper-window success `0.68`
and best success `0.80`, plus task 4 seed 0/1 paper-window success `0.37` and
`0.50`.

Interpretation: Method A is feasible and produces meaningful control signal,
but robustness is limited by late fine-tuning collapse and seed variance.

### Method B: TCN/InfoNCE Cross-Modal InFOM

Evidence package:

- `agents/cross_modal_tcn_infom.py`
- `tests/test_cross_modal_tcn_infom.py`
- `output/jupyter-notebook/method-b-ft-collapse-analysis.executed.ipynb`
- `runs/unity_cross_modal_tcn_method_b/**/manifest.json`

The Method B runs support the negative finding that cross-modal retrieval or
alignment quality alone was not sufficient for reliable control transfer in the
current setup. Several runs completed, but failed/cancelled attempts and
fine-tuning instability remain part of the evidence.

Interpretation: scaling Method B further is not the strongest next experiment
without a new stabilization idea.

## Failure, Retry, And Caveat Ledger

The final scan deliberately includes failed and retried work:

- Early `infom-repro-*` attempts failed quickly on L40S jobs before later
  replacement runs completed.
- Several paper-reproduction and A/B jobs were cancelled before the successful
  A100/H100 replacements.
- `bridge-state-smoke` hit `OUT_OF_MEMORY` at `32G`; a later
  `bridge-state-loader-smoke` completed at `64G`.
- TCN fine-tuning had cancelled and failed pack attempts before later completed
  diagnostic/full runs.
- Demo videos are not included as success evidence because existing summaries
  aggregate over 10 episodes while videos are not individually success-labeled.
- Best-checkpoint success and paper-window success are reported separately.

## Merge Readiness

The final merge package should be reviewed as a compact source-plus-evidence
PR, not as a raw experiment archive. The raw runs remain on Unity/local disk;
the Git package contains the implementation, focused tests, final report,
selected notebooks/figures, concise reports, and manifest provenance.

Residual risk:

- The project evidence is mixed, especially for downstream control robustness.
- Unity's archival clone should be updated only after the final PR is merged.
- Any future paper-facing claim should cite exact paths above and keep
  paper-window metrics separate from best-checkpoint diagnostics.
