# Learning Latent Intent From Observation - Slide Plan

This is a 5-minute academic-style talk plan for an audience familiar with ML/RL but unfamiliar with this project. The posture is conservative: this project produced a working bridge pipeline and promising signals, not a solved benchmark.

Primary notebook: `output/jupyter-notebook/final-project-presentation.ipynb`

## Timing

| Segment | Time | Purpose |
| --- | ---: | --- |
| Motivation and InFOM background | 0:00-1:00 | Explain the goal and give a brief InFOM primer |
| Concrete setting | 1:00-2:00 | Show the state input, RGB bridge, and action interface |
| Methods | 2:00-3:00 | Show our method, the contrastive backup, and what actually runs |
| Metrics and results | 3:00-4:25 | Present the strongest signals and caveats |
| Contributions and takeaway | 4:25-5:00 | End with what the project adds |

## Slide 1 - Motivation: Learning Intent From Observation

**Purpose:** Give the audience the project-level goal before introducing InFOM.

**Visual:** Use `assets/infom.png` as a background/reference image or use a simple text diagram: observation -> latent intent -> control.

**Spoken points:**
- Standard RL learns from scalar reward and trial-and-error.
- This project asks whether trajectories can reveal reusable intent abstractions.
- The intent abstraction should be useful for control, not only for prediction or clustering.

**Report expansion notes:** Define intent as a temporally extended behavioral mode learned from trajectory segments. Explicitly distinguish it from a hand-written goal label, inverse-RL reward function, and demonstrator identity.

## Slide 2 - Brief Background: What InFOM Gives Us

**Purpose:** Introduce the base method without over-explaining it.

**Visual:** Use `output/jupyter-notebook/artifacts/final-project-presentation/infom_brief_pipeline.png`.

**Spoken points:**
- InFOM learns latent intentions from reward-free trajectory data.
- It uses those latents in an intention-conditioned flow occupancy model.
- This project keeps that structure but changes the observation setting.

**Report expansion notes:** Mention the nearby literature framing from the proposal: Third-Person Imitation Learning, ToMnet, and InFOM. The project sits between observational learning and offline RL representation learning.

## Slide 3 - Problem Statement: Cross-View Control Mismatch

**Purpose:** Make the technical challenge concrete.

**Visual:** Three-row pipeline:

```text
Pretraining:    state + third-person RGB + actions
Fine-tuning:    RGB bridge + rewards
Deployment:     state-only control
```

**Spoken points:**
- The core challenge is not just learning a latent.
- The latent has to survive a handoff between observation modalities.
- I use the agent's own synchronized third-person view as the first bridge.

**Report expansion notes:** Explain why this is a staged approximation to the original cross-agent goal. The self third-person bridge isolates viewpoint/modality mismatch while keeping dataset and simulator risk manageable.

## Slide 4 - Concrete Example: What The Policy Sees And Controls

**Purpose:** Make the setup legible to unfamiliar listeners.

**Visual:** Use `output/jupyter-notebook/artifacts/final-project-presentation/obs_action_example.png`.

**Spoken points:**
- One bridge transition contains a 28-D low-dimensional state, a synchronized 64x64 third-person RGB frame, and a 5-D continuous action.
- The deployed actor sees the 28-D state only.
- The actor controls relative end-effector x/y/z, yaw, and gripper opening for a UR5e arm with a Robotiq gripper.
- Saved `qpos` and `qvel` help with replay/rendering/auditing; they are not the exact deployed policy input.

**Report expansion notes:** The 28-D observation contains arm joint positions/velocities, end-effector state, gripper state/contact, and cube pose. This concrete interface explains why preserving state-like control structure matters.

## Slide 5 - Hypotheses

**Purpose:** State what the experiments test.

**Visual:** Three hypothesis boxes.

**Spoken points:**
- H1: paired state/RGB trajectories can align two views into a shared latent.
- H2: that shared latent can support downstream state-only control.
- H3: alignment and action-consistency diagnostics explain success or failure.

**Report expansion notes:** The final evidence separates H1 and H2. The contrastive backup supports H1 but weakens the simple version of H2, because excellent retrieval alignment did not produce strong control transfer.

## Slide 6 - Method: State-Distilled InFOM

**Purpose:** Focus the talk on the best-supported method while keeping the contrastive backup as a short ablation.

**Visual:** Small component table for our method.

| Component | Role |
| --- | --- |
| Fixed state encoder | normalizes the 28-D low-dimensional state |
| RGB bridge | maps third-person RGB into the same 28-D latent |
| Bridge loss | supervised RGB-to-state MSE |
| InFOM backbone | intent encoder, flow occupancy, reward, critic, actor |
| Evaluation | state-only actor, no RGB |

**Spoken points:**
- Our method is the approach I would present as the main contribution.
- It is deliberately state-like because the actor and critic need control-relevant structure.
- The contrastive backup is a useful negative ablation, but not a main presentation plot.

**Report expansion notes:** Our method also trains actor behavior cloning on both RGB-derived and true-state latents. The contrastive backup used unit-normalized embeddings and in-batch negatives; it showed that strong cross-modal retrieval is not sufficient for control.

## Slide 7 - What Actually Runs

**Purpose:** Answer the implementation-specific question: what algorithm is running besides "InFOM"?

**Visual:** Use `output/jupyter-notebook/artifacts/final-project-presentation/method_a_algorithm_pipeline.png`.

**Spoken points:**
- Our method adds a state normalizer, `impala_small` RGB encoder, RGB-to-state MSE bridge loss, and extra behavior cloning on true-state latents.
- The InFOM backbone still supplies intention encoding, flow occupancy modeling, reward prediction, critic learning, and actor updates.
- Fine-tuning trains reward/critic/actor/flow components on reward-labeled data while keeping the bridge stable.
- Evaluation discards RGB and runs `state -> normalizer -> actor -> 5-D action`.

**Report expansion notes:** The contrastive backup replaces the MSE state-distillation bridge with a symmetric InfoNCE bridge between state and RGB encoders. Its strong retrieval diagnostics but weak control results are central to the interpretation.

## Slide 8 - Metrics: More Than Success Rate

**Purpose:** Answer why success rate alone is insufficient.

**Visual:** Metric table or compact dashboard.

**Spoken points:**
- Paper-window success is the conservative headline metric.
- Best success and collapse show whether training finds behavior and then loses it.
- Return and episode length capture partial progress.
- RGB-to-state alignment loss and RGB-state action agreement diagnose our bridge.
- Demo summaries are aggregate context only; do not show the current MP4/GIFs as success evidence because they are not individually success-labeled.

**Report expansion notes:** Keep best-checkpoint results separate from paper-window metrics. The key diagnostic is how much late-checkpoint performance drops from the best checkpoint.

## Slide 9 - Results: Promising But Not Clean

**Purpose:** Present the main empirical result without overstating it.

**Visuals:**
- `output/jupyter-notebook/artifacts/final-project-presentation/method_a_task1_mean_success_curve.png`
- `output/jupyter-notebook/artifacts/final-project-presentation/method_a_best_vs_paper.png`
- `output/jupyter-notebook/artifacts/final-project-presentation/method_a_success_curves.png` for per-seed backup

**Spoken points:**
- Our method is the strongest bridge method in the current package.
- Task1 over five seeds: paper-window success 0.191, mean best success 0.656, best seed 0.800.
- Task4 over six seeds: paper-window success 0.199, mean best success 0.477, best seed 0.660.
- The method often finds behavior, but late fine-tuning collapse makes paper-window results much weaker than best checkpoints.
- This is useful signal, not a benchmark win.

**Report expansion notes:** Include the task-level table from `output/jupyter-notebook/artifacts/final-project-presentation/method_a_presentation_summary.csv`.

## Slide 10 - What The Pieces Learned

**Purpose:** Demonstrate that the bridge and InFOM pieces learned useful structure before discussing final collapse.

**Visuals:**
- `output/jupyter-notebook/artifacts/final-project-presentation/method_a_bridge_learning.png`
- `output/jupyter-notebook/artifacts/final-project-presentation/method_a_infom_intent_learning.png`

**Spoken points:**
- The RGB-to-state bridge loss drops during pretraining.
- RGB/state actor agreement improves, especially action cosine.
- InFOM flow/intent losses drop after bridge warmup.
- The failure mode is therefore not "nothing learned"; it is robustness during late fine-tuning.

**Report expansion notes:** This is the strongest conceptual contribution: component diagnostics can be positive while final offline RL performance remains unstable. The bridge and intent model learn useful structure, but paper-window success is limited by collapse and seed variance.

## Slide 11 - Contributions And Limitations

**Purpose:** Package the project as a credible research contribution.

**Visual:** Two-column table: contributions vs limitations.

**Spoken points:**
- Contributions: problem framing, bridge dataset/protocol, our state-distilled method, the contrastive backup, and evidence audit notebook.
- Limitations: limited seeds, late collapse, self-bridge rather than true cross-agent transfer, paper replication caveats.
- The honest conclusion is that the project found a viable path and important failure modes.

**Report expansion notes:** Mention the earlier matched bridge-aware run as mixed preliminary evidence: late-window success improved over ego-only in that run, but best-over-training did not clearly favor bridge-aware InFOM.

## Slide 12 - Closing Takeaway

**Purpose:** End with the broad goal and how the results relate to it.

**Visual:** One-sentence takeaway over a simple pipeline diagram.

**Spoken points:**
- The overarching goal is observation-grounded behavioral abstraction for control.
- This project shows that the bridge can be built and can produce real control signal.
- The main open problem is robustness: preserving control-relevant structure through fine-tuning.

**Report expansion notes:** The strongest next experiment is our method anti-collapse work: smaller FT learning rate, stronger conservative/BC regularization, reward/critic stabilization, checkpoint selection, and explicit paper-window versus best-checkpoint reporting.

## Demo Slide Decision

Do not include the current demo videos in the 5-minute presentation.

The summary JSON reports aggregate success across 10 evaluation episodes, but the renderer saved only the first two video episodes and does not label each saved video as success or failure. Since the visible videos do not clearly show success, use the our method metric plots instead.

To use demos later, regenerate them with per-video success metadata, visible success/failure labels, and a selected successful rollout.

## Presentation Rule

Do not say "our method solves cross-modal intent transfer." Say:

> Our method gives the strongest evidence that a state-like cross-modal bridge can produce usable control signal, but the result is task-dependent and unstable.
