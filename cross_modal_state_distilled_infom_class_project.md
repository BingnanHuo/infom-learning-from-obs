# Cross-Modal State-Distilled InFOM

**Target:** class-project implementation spec for coding agent
**Immediate goal:** get a simple working version ASAP
**Main method to implement first:** RGB-to-state latent distillation + InFOM
**Backup method:** TCN-style contrastive alignment
**Assumed dataset status:** paired bridge data already exists: each transition has both low-dimensional state and RGB render image.

---

## 0. One-paragraph project idea

We want to keep the project mainly about **InFOM-style reward-free offline RL pretraining**. The twist is a cross-modal observation mismatch: during pretraining, we have both low-dimensional state observations and RGB images; during reward-labeled fine-tuning, we only use RGB observations, actions, next RGB observations, and rewards; at evaluation/deployment, the policy acts from low-dimensional state observations. To make this possible, we train the RGB encoder and state encoder to produce compatible latent inputs for InFOM. The first implementation should use a simple and robust **state-as-latent distillation** bridge: train the RGB encoder to predict the normalized low-dimensional state, then run InFOM in this state-like latent space.

---

## 1. What to implement first

Implement **Method A: State-Distilled InFOM** first.

Do **not** start with a fully general shared 512-dimensional contrastive latent unless Method A fails. Method A is less elegant, but much easier to debug and much more likely to work in a short class-project timeline.

Recommended first experiment:

```text
Domain: cube-single
Task: task1
Pretraining data: paired state + RGB reward-free play data
Fine-tuning data: RGB-only reward-labeled transitions
Evaluation: state-only observations
```

---

## 2. Original InFOM pieces we should keep

Original InFOM has this structure:

```text
Reward-free offline pretraining:
    train intention encoder + flow occupancy model + BC policy

Reward-labeled offline fine-tuning:
    train reward predictor + critic + actor using generated future states/latents

Evaluation:
    use actor policy
```

The original paper uses an unlabeled reward-free transition dataset for pretraining and a reward-labeled offline dataset for fine-tuning. For OGBench, the paper reports 1M unlabeled play transitions for pretraining and 500K reward-labeled transitions for fine-tuning. For image-based OGBench tasks, it uses a small IMPALA encoder, random cropping augmentation with probability 0.5, and frame stacking with three images.

The official repo already contains:

```text
utils/encoders.py:
    mlp
    impala
    impala_small
    impala_large
    resnet_34

utils/networks.py:
    MLP
    Value
    Actor
    IntentionEncoder
    VectorField

agents/infom.py:
    InFOM pretraining and fine-tuning losses
    actor loss with BC regularization
    flow-goal generation
    target vector field updates

utils/datasets.py:
    frame stacking
    random crop image augmentation
    next_actions support
```

For this project, reuse as much of the original InFOM code as possible. The main new code should be the cross-modal bridge and the modified batch encoding logic.

---

## 3. Data assumptions and required batch keys

### 3.1 Pretraining bridge dataset

The bridge dataset is assumed to be complete and paired:

```python
batch_pre = {
    "states": low_dim_state_t,                 # [B, state_dim]
    "next_states": low_dim_state_t1,           # [B, state_dim]

    "observations_rgb": rgb_t,                 # [B, H, W, C * frame_stack]
    "next_observations_rgb": rgb_t1,           # [B, H, W, C * frame_stack]

    "actions": a_t,                            # [B, action_dim]
    "next_actions": a_t1,                      # [B, action_dim]

    "terminals": done_t,
    "masks": mask_t,
}
```

The pairing must be exact:

```text
states[i] and observations_rgb[i] must describe the same timestep.
next_states[i] and next_observations_rgb[i] must describe the same next timestep.
```

### 3.2 Fine-tuning RGB reward dataset

Fine-tuning is RGB-only, but still needs actions and next observations for offline RL/InFOM:

```python
batch_ft = {
    "observations_rgb": rgb_t,
    "next_observations_rgb": rgb_t1,
    "actions": a_t,
    "next_actions": a_t1,
    "rewards": r_t,
    "terminals": done_t,
    "masks": mask_t,
}
```

Important: if there are no actions and next observations, this is not standard offline RL or InFOM fine-tuning.

### 3.3 Evaluation data

Evaluation/deployment uses only low-dimensional state:

```python
obs_eval = {
    "states": low_dim_state_t
}
```

Policy call:

```python
h = E_state(states)
a = actor(h)
```

No RGB should be used at evaluation for this experiment.

---

# Method A: State-Distilled InFOM, primary method

## 4. Core idea

Use the normalized low-dimensional state as the shared latent space.

Let:

\[
s_t \in \mathbb{R}^{d_s}
\]

be the low-dimensional state, and:

\[
x_t
\]

be the RGB observation or RGB frame stack.

Define:

\[
h_t^{state} = E_s(s_t)
\]

and:

\[
h_t^{rgb} = E_x(x_t).
\]

For the first implementation:

\[
E_s(s_t) = \mathrm{Normalize}(s_t)
\]

and:

\[
E_x(x_t) \approx \mathrm{Normalize}(s_t).
\]

So the shared latent dimension is:

\[
d_h = d_s.
\]

This means the RGB encoder outputs a **pseudo-state**, not a generic hidden vector.

---

## 5. Why this is the recommended first method

This uses the generous assumption that paired state/RGB data exist during pretraining. It avoids fragile choices around contrastive negatives, temperature tuning, adversarial domain confusion, and representation collapse. It is also easy to test: the RGB encoder should reconstruct normalized state from images.

The project story remains InFOM-centered:

```text
1. Learn a bridge from RGB to the state-like latent space.
2. Pretrain InFOM's intention-conditioned flow occupancy model in that latent space.
3. Fine-tune InFOM using RGB reward-labeled data encoded into pseudo-state latents.
4. Deploy the same actor using true state latents.
```

This is a practical bridge, not a cathedral. Build the bridge first. Add gargoyles later. 🏗️

---

## 6. Encoders for Method A

### 6.1 State encoder `E_state`

Use a fixed normalizer, not a trainable MLP, for the first implementation.

```python
class StateNormalizer:
    def __init__(self, mean, std, eps=1e-8):
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, s):
        return (s - self.mean) / (self.std + self.eps)
```

Compute `mean` and `std` from the paired reward-free pretraining states.

Recommended clipping for stability:

```python
h_state = clip((s - mean) / (std + eps), -5.0, 5.0)
```

Also store:

```python
h_min = min(h_state over pretraining data)
h_max = max(h_state over pretraining data)
```

Use these as the latent bounds for flow-goal clipping if the original InFOM code expects `observation_min` and `observation_max`.

### 6.2 RGB encoder `E_rgb`

Use the repo's existing `impala_small` encoder as the visual trunk.

Original repo encoder details:

```text
encoder_modules['impala_small'] = ImpalaEncoder(num_blocks=1)
```

The default `ImpalaEncoder` uses:

```text
stack_sizes = (16, 32, 32)
width = 1
num_blocks = 1 for impala_small
mlp_hidden_dims = (512,)
input normalization = image / 255.0
activation = ReLU in the conv trunk
output = 512-dimensional feature by default
```

For Method A, add a small projection head:

```python
rgb_feat = impala_small(x_rgb)        # [B, 512]
h_rgb = Linear(state_dim)(rgb_feat)   # [B, state_dim]
```

Recommended optional head:

```text
Linear(512 -> 512), GELU, LayerNorm, Linear(512 -> state_dim)
```

Start with the simpler version first:

```text
IMPALA small -> Linear(state_dim)
```

### 6.3 RGB input format

Use frame stacking for RGB, matching original visual InFOM:

```text
frame_stack = 3
p_aug = 0.5
random crop padding = use repo default
```

For OGBench visual observations, this usually means:

```text
x_rgb shape: [B, 64, 64, 9]
```

if images are stacked along channels.

---

## 7. Method A losses

### 7.1 RGB-to-state alignment loss

For paired pretraining data:

\[
h_t^{state} = E_s(s_t), \qquad h_t^{rgb} = E_x(x_t).
\]

Use mean squared error or Huber loss:

\[
\mathcal{L}_{align}
= \left\|h_t^{rgb} - \mathrm{stopgrad}(h_t^{state})\right\|_2^2.
\]

Also apply it to next observations:

\[
\mathcal{L}_{align-next}
= \left\|E_x(x_{t+1}) - \mathrm{stopgrad}(E_s(s_{t+1}))\right\|_2^2.
\]

Recommended combined loss:

\[
\mathcal{L}_{align-total}
= \mathcal{L}_{align} + \mathcal{L}_{align-next}.
\]

Suggested weight:

```python
lambda_align = 1.0
```

If this dominates and the InFOM losses do not train, reduce to:

```python
lambda_align = 0.1
```

### 7.2 InFOM pretraining loss in pseudo-state latent space

Use RGB as the canonical pretraining view because fine-tuning will only have RGB.

```python
h = E_rgb(observations_rgb)
h_next = E_rgb(next_observations_rgb)
```

Then feed these into the usual InFOM pretraining losses as if they were observations:

```python
infom_batch = {
    "observations": h,
    "next_observations": h_next,
    "actions": actions,
    "next_actions": next_actions,
    "terminals": terminals,
    "masks": masks,
    "observation_min": h_min,
    "observation_max": h_max,
}
```

The original InFOM intention encoder should infer the intention latent from the next latent/action pair:

\[
\eta_t \sim q_\phi(\eta \mid h_{t+1}, a_{t+1}).
\]

The flow occupancy model predicts future latent states:

\[
q_\theta(h_f \mid h_t, a_t, \eta_t).
\]

### 7.3 Behavior cloning on both modalities

This is a small addition that directly helps the deployment mismatch.

Use the same actor on both latent sources:

\[
\mathcal{L}_{BC-view}
= -\log \pi(a_t \mid h_t^{rgb})
  -\log \pi(a_t \mid h_t^{state}).
\]

The first term teaches the actor to act from RGB-derived pseudo-state latents. The second term teaches it to act from true state latents, which are what it will receive at evaluation time.

Suggested weight:

```python
lambda_bc_view = 1.0
```

If original InFOM already has a BC term, either:

```text
Option 1: replace original BC with BC-view
Option 2: keep original BC on h_rgb and add extra BC on h_state
```

Recommended: **Option 2**, because it is a minimal patch.

### 7.4 Total pretraining objective

Use:

\[
\boxed{
\mathcal{L}_{pre}
= \mathcal{L}_{InFOM-pre}(h^{rgb})
+ \lambda_{align}\mathcal{L}_{align-total}
+ \lambda_{bc-view}\mathcal{L}_{BC-state-extra}
}
\]

where `L_InFOM-pre` includes the original flow occupancy loss and original RGB-latent BC loss.

### 7.5 Recommended warm-up trick

Before joint InFOM pretraining, do a short supervised bridge warm-up:

```text
warmup_steps = 10k to 50k
optimize only E_rgb using L_align-total
```

Then run joint pretraining:

```text
joint_pretraining_steps = original visual setting or smaller class-project budget
```

This makes the rest of InFOM see a less chaotic latent space. It is one of the few extra tricks worth adding because it is simple and likely stabilizing.

---

## 8. Method A fine-tuning

Fine-tuning data only has RGB + reward-labeled transitions:

\[
D_{ft}^{rgb} = \{x_t, a_t, x_{t+1}, a_{t+1}, r_t, done_t\}.
\]

Encode:

\[
h_t = E_x(x_t), \qquad h_{t+1} = E_x(x_{t+1}).
\]

Then run the original InFOM fine-tuning losses on:

```python
infom_batch_ft = {
    "observations": h,
    "next_observations": h_next,
    "actions": actions,
    "next_actions": next_actions,
    "rewards": rewards,
    "terminals": terminals,
    "masks": masks,
    "observation_min": h_min,
    "observation_max": h_max,
}
```

### 8.1 Freeze `E_rgb` during first fine-tuning attempt

For the first implementation:

```python
freeze_rgb_encoder_during_finetune = True
freeze_state_encoder_during_finetune = True
```

Why:

During fine-tuning, only RGB data are available. If `E_rgb` keeps changing while `E_state` is fixed, then the actor/critic/reward model may adapt to a new RGB latent distribution that no longer matches the state latent distribution. Then evaluation from state can fail even if fine-tuning metrics look good.

In plain terms: freezing keeps the bridge from moving while the robot is crossing it.

### 8.2 What to train during fine-tuning

Train:

```text
reward predictor r(h)
critic Q(h, a)
actor pi(a | h)
flow vector field
intention encoder
```

Freeze:

```text
E_rgb
E_state / state normalizer
```

If performance is poor and alignment diagnostics are good, a later ablation can unfreeze `E_rgb` with a drift penalty. Do not do this first.

---

## 9. Method A evaluation

At evaluation/deployment:

```python
h = E_state(state_obs)
a = actor(h)
```

No RGB input should be used.

Also test action consistency on paired validation data:

```python
h_state = E_state(states)
h_rgb = E_rgb(observations_rgb)
a_state = actor.mode(h_state)
a_rgb = actor.mode(h_rgb)
consistency_mse = mean((a_state - a_rgb) ** 2)
```

This is a very useful low-level diagnostic. If this number is large, the policy will probably fail in state-only evaluation.

---

# Method B: TCN-style contrastive alignment, backup method

## 10. When to use Method B

Use Method B only if Method A is clearly failing because RGB cannot accurately predict the full low-dimensional state.

Examples:

```text
state contains hidden simulator variables that are not visible in RGB
state includes velocities that cannot be inferred even with frame_stack=3
RGB-to-state validation MSE stays high and action consistency is poor
```

If Method A is working, do not switch. A working simple method beats a glamorous broken one.

---

## 11. Core idea

Instead of forcing RGB to reconstruct the exact normalized state, learn a shared embedding:

\[
h_t^{state} = E_s(s_t) \in \mathbb{R}^{512},
\]

\[
h_t^{rgb} = E_x(x_t) \in \mathbb{R}^{512}.
\]

Use paired same-timestep state/RGB as positives and other batch elements as negatives:

\[
(s_t, x_t) \text{ is positive},
\]

\[
(s_t, x_j), j \neq t \text{ are negatives}.
\]

This is inspired by Time-Contrastive Networks: simultaneous viewpoints of the same moment are pulled together, while different moments are pushed apart. For our setting, the two “views” are not two cameras; they are low-dimensional state and RGB image. The spirit is still useful: learn what is common across modalities and changes across time.

---

## 12. Method B encoders

Use:

```text
E_state: repo MLP encoder -> 512 dim
E_rgb: impala_small -> 512 dim
```

Normalize embeddings before InfoNCE:

```python
h_state = h_state / (norm(h_state, axis=-1, keepdims=True) + 1e-8)
h_rgb = h_rgb / (norm(h_rgb, axis=-1, keepdims=True) + 1e-8)
```

---

## 13. Method B alignment loss

Symmetric InfoNCE:

\[
\mathcal{L}_{s \rightarrow x}
= -\frac{1}{B}\sum_i
\log
\frac{
\exp(\mathrm{sim}(h_i^{state}, h_i^{rgb})/\tau)
}{
\sum_j \exp(\mathrm{sim}(h_i^{state}, h_j^{rgb})/\tau)
},
\]

\[
\mathcal{L}_{x \rightarrow s}
= -\frac{1}{B}\sum_i
\log
\frac{
\exp(\mathrm{sim}(h_i^{rgb}, h_i^{state})/\tau)
}{
\sum_j \exp(\mathrm{sim}(h_i^{rgb}, h_j^{state})/\tau)
}.
\]

Total:

\[
\mathcal{L}_{InfoNCE}
= \mathcal{L}_{s \rightarrow x} + \mathcal{L}_{x \rightarrow s}.
\]

Suggested params:

```python
h_dim = 512
temperature = 0.1
lambda_infonce = 1.0
batch_size = as large as memory allows, ideally >= 256
```

Do not add hard negatives initially. Use in-batch negatives first.

---

## 14. Method B with InFOM

InFOM then runs on the 512-dimensional shared embedding:

```python
h = E_rgb(rgb)
h_next = E_rgb(rgb_next)
```

for pretraining/fine-tuning, and:

```python
h = E_state(state)
```

for evaluation.

This requires modifying InFOM's observation dimension to `h_dim = 512`.

Method B is more general, but harder to debug. Keep it as a fallback or extra experiment.

---

# Third-Person Imitation Learning inspiration

## 15. What to borrow and what not to borrow

Third-Person Imitation Learning is relevant because it tries to learn features that are invariant to viewpoint/domain differences. Its domain-confusion idea is conceptually useful, but not necessary for our first implementation because we have paired state/RGB data.

For this project, borrow the motivation:

```text
The latent should not reveal which modality produced it.
```

Do **not** implement adversarial domain confusion first. It adds fragility and is weaker than direct paired supervision in our setting.

### Lightweight diagnostic inspired by TPIL

Train a small post-hoc classifier:

```python
view_classifier(h) -> {state, rgb}
```

using frozen latents from paired validation data.

Interpretation:

```text
If classifier accuracy is near 50%: modality alignment is good.
If classifier accuracy is very high: state and RGB latents are still easy to distinguish.
```

For Method A, the classifier may still detect small distribution differences because `h_state` is exact normalized state and `h_rgb` is predicted pseudo-state. That is okay. More important diagnostics are alignment MSE and actor action consistency.

---

# Recommended configs

## 16. Main config for Method A

```python
agent_name = "cross_modal_state_distilled_infom"

bridge_method = "state_distillation"

# Data
pretrain_view = "paired_state_rgb"
finetune_view = "rgb_only"
eval_view = "state_only"
frame_stack = 3
p_aug = 0.5

# Encoders
state_encoder = "normalizer"          # fixed, not trainable
rgb_encoder = "impala_small"
rgb_projection_dim = state_dim

# Distillation
lambda_align = 1.0
align_loss = "mse"                    # or "huber"
state_norm = "standard"               # (s - mean) / std
state_clip = 5.0
warmup_align_steps = 10000             # increase to 50000 if unstable

# Actor modality support
lambda_bc_state_extra = 1.0
bc_on_rgb_latent = True
bc_on_state_latent = True

# Fine-tuning
freeze_rgb_encoder_during_finetune = True
freeze_state_encoder_during_finetune = True

# InFOM, keep close to repo defaults first
latent_intention_dim = 512             # InFOM eta dimension; separate from h dimension
num_flow_goals = 16
num_flow_steps = 10
clip_flow_goals = True
```

Note the naming:

```text
h = shared observation latent / pseudo-state
eta = InFOM intention latent
```

Do not call both `z`.

## 17. Backup config for Method B

```python
bridge_method = "infonce"

state_encoder = "mlp"                 # repo MLP, output 512
rgb_encoder = "impala_small"          # output 512
shared_latent_dim = 512
normalize_embeddings = True
infonce_temperature = 0.1
lambda_infonce = 1.0
batch_size = 256_or_larger_if_possible
freeze_encoders_during_finetune = True
```

---

# Minimal code changes

## 18. Add or adapt agent

Suggested file:

```text
agents/cross_modal_state_distilled_infom.py
```

Start from:

```text
agents/infom.py
```

Add:

```python
encode_state(states)
encode_rgb(rgb)
make_infom_batch_from_latents(...)
alignment_loss(...)
pretraining_loss_cross_modal(...)
finetuning_loss_rgb_only(...)
sample_actions_state_only(...)
```

## 19. Batch conversion helpers

### Pretraining helper

```python
def make_pretrain_latent_batch(batch):
    h_state = encode_state(batch["states"])
    h_next_state = encode_state(batch["next_states"])

    h_rgb = encode_rgb(batch["observations_rgb"])
    h_next_rgb = encode_rgb(batch["next_observations_rgb"])

    infom_batch = {
        "observations": h_rgb,
        "next_observations": h_next_rgb,
        "actions": batch["actions"],
        "next_actions": batch["next_actions"],
        "terminals": batch["terminals"],
        "masks": batch["masks"],
        "observation_min": h_min,
        "observation_max": h_max,
    }
    return infom_batch, h_state, h_next_state, h_rgb, h_next_rgb
```

### Fine-tuning helper

```python
def make_finetune_latent_batch(batch):
    h_rgb = stop_grad_if_frozen(encode_rgb(batch["observations_rgb"]))
    h_next_rgb = stop_grad_if_frozen(encode_rgb(batch["next_observations_rgb"]))

    infom_batch = {
        "observations": h_rgb,
        "next_observations": h_next_rgb,
        "actions": batch["actions"],
        "next_actions": batch["next_actions"],
        "rewards": batch["rewards"],
        "terminals": batch["terminals"],
        "masks": batch["masks"],
        "observation_min": h_min,
        "observation_max": h_max,
    }
    return infom_batch
```

### Evaluation helper

```python
def sample_actions_from_state(states, rng):
    h_state = encode_state(states)
    return actor.sample_actions(h_state, rng)
```

---

# Low-level tests and diagnostics

The high-level success-rate experiment is not enough. Add these module-wise tests so we know which part is broken if results are poor.

## 20. Dataset pairing tests

### Test 1: shape and dtype sanity

Check one batch:

```text
states: [B, state_dim], float32
next_states: [B, state_dim], float32
observations_rgb: [B, 64, 64, 3 * frame_stack], uint8 or float32
next_observations_rgb: same as observations_rgb
actions: [B, action_dim], float32
next_actions: [B, action_dim], float32
```

### Test 2: paired timestep visual sanity

Randomly save a grid of:

```text
state summary + RGB image
next_state summary + next RGB image
```

For cube tasks, at minimum print object/gripper coordinates from state next to the RGB image. This catches accidental dataset misalignment.

### Test 3: next-action sanity

Check:

```python
next_actions[t] approximately equals actions[t+1]
```

except at episode boundaries. The original repo warns that next-action logic can be incorrect at trajectory ends if handled carelessly, so masks/terminals matter.

---

## 21. State normalizer tests

After computing state statistics:

```python
h_state = E_state(states_val)
```

Log:

```text
mean per dimension should be near 0
std per dimension should be near 1
min/max should be finite
no NaNs
```

Also save `h_min` and `h_max` for flow clipping.

---

## 22. RGB encoder distillation tests

### Test 1: overfit tiny batch

Before full training, overfit `E_rgb` on 128 paired samples.

Expected:

```text
alignment MSE should go close to 0 or decrease dramatically
```

If it cannot overfit a tiny batch, the encoder/loss/data wiring is broken.

### Test 2: train/validation MSE

During warm-up, log:

```text
align_mse_train
align_mse_val
align_mse_next_train
align_mse_next_val
```

Expected:

```text
both should decrease
validation should not be totally flat
```

### Test 3: dimension-wise MSE

Some state dimensions may be impossible to infer from RGB. Log per-dimension MSE:

```python
mse_per_dim = mean((h_rgb - h_state) ** 2, axis=0)
```

If a few dimensions dominate, consider dropping those dimensions from the distilled latent or using frame stacking/history.

### Test 4: nearest-neighbor retrieval

On a validation batch:

```text
For each h_rgb[i], find nearest h_state[j].
Top-1 accuracy should be above random.
Mean rank should improve during training.
```

For Method A, this is less important than MSE, but it is a nice alignment sanity check.

---

## 23. Actor modality tests

### Test 1: BC loss by modality

During pretraining, log separately:

```text
bc_loss_rgb
bc_loss_state
bc_mse_action_rgb
bc_mse_action_state
```

Both should improve. If `bc_loss_state` is bad, the actor will likely fail at state-only evaluation.

### Test 2: action consistency across modalities

On paired validation data:

```python
a_rgb = actor.mode(E_rgb(rgb))
a_state = actor.mode(E_state(state))
a_mse = mean((a_rgb - a_state) ** 2)
cos = cosine_similarity(a_rgb, a_state)
```

Expected:

```text
action MSE should decrease during pretraining
```

This is one of the most important diagnostics.

---

## 24. Fine-tuning freeze tests

When `freeze_rgb_encoder_during_finetune=True`, verify the RGB encoder params do not change.

Before fine-tuning:

```python
params_before = tree_copy(E_rgb_params)
```

After N updates:

```python
max_abs_delta = max_abs(params_after - params_before)
```

Expected:

```text
max_abs_delta == 0 or numerically tiny
```

Also verify actor/critic/reward/flow params do change.

---

## 25. Reward predictor tests

During RGB-only fine-tuning:

```text
reward_loss_train
reward_loss_val
pred_reward_mean
true_reward_mean
pred_reward_std
true_reward_std
corr(pred_reward, true_reward)
```

Expected:

```text
reward loss decreases
predicted rewards are not constant
correlation should become positive if rewards are not too sparse
```

If reward prediction is flat, the critic and actor are training on smoke signals.

---

## 26. InFOM flow tests

Log original InFOM flow metrics plus:

```text
flow_loss
latent_goal_mean/std
latent_goal_min/max
percent_generated_latents_outside_h_bounds
```

Expected:

```text
generated h_f should stay mostly within h_min/h_max if clipping is enabled
flow loss should not explode
no NaNs
```

Also compare generated future latent norms to dataset latent norms:

```python
norm_generated = mean(norm(h_future_generated))
norm_data = mean(norm(h_rgb_data))
```

They should be in the same rough range.

---

## 27. Smoke tests before full runs

Run these before any expensive experiment:

```text
1. Data loader returns correct paired batch.
2. E_rgb warm-up overfits tiny batch.
3. Joint pretraining runs 1k updates without NaNs.
4. Fine-tuning runs 1k updates with E_rgb frozen.
5. sample_actions_from_state returns actions with correct shape/range.
6. action consistency between RGB and state latents is not random garbage.
```

---

# Minimal ablations

For the class project, keep ablations small.

## 28. Required runs

### Run 1: original state InFOM oracle-ish baseline

```text
pretrain: state play data
fine-tune: state reward data
evaluate: state
```

### Run 2: original visual InFOM sanity baseline

```text
pretrain: RGB play data
fine-tune: RGB reward data
evaluate: RGB
```

### Run 3: our method, Method A

```text
pretrain: paired state+RGB reward-free data
fine-tune: RGB reward-labeled data
evaluate: state
```

### Run 4: our method without alignment

```text
same as Run 3, but lambda_align = 0
```

This tests whether the bridge matters.

## 29. Optional if time remains

```text
Method A without BC on state latents
Method A with E_rgb unfrozen during fine-tuning
Method B InfoNCE alignment
```

Do not start with these.

---

# Brief related-work positioning

This project is primarily an extension of InFOM to a cross-modal observation mismatch setting. InFOM already studies reward-free pretraining followed by reward-labeled offline fine-tuning, and it already supports visual OGBench with an IMPALA encoder. Our new piece is that the reward-labeled fine-tuning view and deployment view are different.

The primary bridge method is inspired by privileged-information distillation: use structured state as a clean teacher signal for the visual encoder. The contrastive backup is inspired by Time-Contrastive Networks, which align simultaneous observations across views using metric learning. Third-Person Imitation Learning is useful as conceptual motivation for modality-invariant features, but we should not implement adversarial domain confusion in the first version because direct paired supervision is simpler and stronger for our data.

Useful references:

- InFOM paper: <https://arxiv.org/html/2506.08902v3>
- InFOM official repo: <https://github.com/chongyi-zheng/infom>
- InFOM encoder code: <https://raw.githubusercontent.com/chongyi-zheng/infom/main/utils/encoders.py>
- Time-Contrastive Networks: <https://arxiv.org/abs/1704.06888>
- Third-Person Imitation Learning: <https://arxiv.org/abs/1703.01703>
- Learning by Cheating: <https://arxiv.org/abs/1912.12294>

---

# Coding-agent instruction block

Please adapt the previous bridge/shared-latent implementation into the simpler **State-Distilled InFOM** method described here. The dataset bridge is already available with paired low-dimensional state and RGB render image data. First implement Method A only: `E_state = fixed state normalizer`, `E_rgb = impala_small + linear projection to state_dim`, supervised RGB-to-normalized-state alignment loss, InFOM pretraining on RGB-derived pseudo-state latents, extra BC on true state latents, RGB-only reward fine-tuning with `E_rgb` frozen, and state-only evaluation through `E_state`. Add the low-level module tests in Sections 20–27 before running full success-rate experiments. Keep Method B / InfoNCE as a backup, not the first implementation.
