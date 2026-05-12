import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agents.cross_modal_tcn_infom import (
    CrossModalTCNInFOMAgent,
    get_config,
    symmetric_infonce_loss,
)
from utils.datasets import Dataset


def _state_stats(observations, shared_latent_dim, state_clip=5.0, eps=1e-8):
    observations = observations.astype(np.float32)
    mean = observations.mean(axis=0)
    std = np.sqrt(observations.var(axis=0) + eps)
    return dict(
        state_mean=mean,
        state_std=std,
        latent_min=-np.ones((shared_latent_dim,), dtype=np.float32),
        latent_max=np.ones((shared_latent_dim,), dtype=np.float32),
    )


def _tree_max_abs_delta(a, b):
    leaves = jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda x, y: jnp.max(jnp.abs(x - y)), a, b))
    return float(jnp.max(jnp.stack(leaves)))


def _tiny_config():
    config = get_config()
    config.lr = 1e-3
    config.batch_size = 4
    config.rgb_encoder = 'impala_debug'
    config.rgb_frame_stack = 3
    config.rgb_p_aug = None
    config.shared_latent_dim = 16
    config.state_encoder_hidden_dims = (16, 16)
    config.warmup_align_steps = 0
    config.latent_dim = 8
    config.num_flow_goals = 2
    config.num_flow_steps = 2
    config.intention_encoder_hidden_dims = (16, 16)
    config.actor_hidden_dims = (16, 16)
    config.value_hidden_dims = (16, 16)
    config.reward_hidden_dims = (16, 16)
    return config


def _synthetic_dataset(num_steps=12, state_dim=5, action_dim=2, image_size=16):
    rng = np.random.default_rng(123)
    observations = rng.normal(size=(num_steps, state_dim)).astype(np.float32)
    next_observations = np.roll(observations, -1, axis=0)
    actions = np.linspace(-0.8, 0.8, num_steps * action_dim, dtype=np.float32).reshape(num_steps, action_dim)
    rgb = rng.integers(0, 255, size=(num_steps, image_size, image_size, 3), dtype=np.uint8)
    next_rgb = np.roll(rgb, -1, axis=0)
    rewards = rng.normal(size=(num_steps,)).astype(np.float32)
    terminals = np.zeros((num_steps,), dtype=np.float32)
    terminals[[5, 11]] = 1.0

    dataset = Dataset.create(
        observations=observations,
        next_observations=next_observations,
        third_person_observations=rgb,
        next_third_person_observations=next_rgb,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        masks=1.0 - terminals,
        freeze=False,
    )
    dataset.obs_norm_type = 'none'
    dataset.return_next_actions = True
    dataset.aux_frame_stack = 3
    dataset.aux_frame_stack_keys = ('third_person_observations',)
    dataset.aug_keys = ('third_person_observations', 'next_third_person_observations')
    dataset.normalize_observations()
    return dataset


def _make_agent_and_batch():
    config = _tiny_config()
    dataset = _synthetic_dataset()
    batch = dataset.sample(config.batch_size, idxs=np.array([0, 1, 6, 7]))
    stats = _state_stats(
        dataset['observations'],
        shared_latent_dim=config.shared_latent_dim,
        state_clip=config.state_clip,
        eps=config.state_norm_eps,
    )
    agent = CrossModalTCNInFOMAgent.create(
        seed=0,
        ex_observations=batch['observations'],
        ex_actions=batch['actions'],
        config=config,
        ex_third_person_observations=batch['third_person_observations'],
        **stats,
    )
    return agent, batch


def test_symmetric_infonce_prefers_matched_pairs():
    embeddings = jnp.eye(4, dtype=jnp.float32)
    matched_loss, matched_info = symmetric_infonce_loss(embeddings, embeddings, temperature=0.1)
    shifted_loss, shifted_info = symmetric_infonce_loss(embeddings, jnp.roll(embeddings, 1, axis=0), temperature=0.1)

    assert float(matched_loss) < float(shifted_loss)
    assert float(matched_info['retrieval_acc_state_to_rgb']) == 1.0
    assert float(matched_info['retrieval_acc_rgb_to_state']) == 1.0
    assert float(shifted_info['retrieval_acc_state_to_rgb']) == 0.0


def test_state_rgb_encoders_emit_unit_shared_latents():
    agent, batch = _make_agent_and_batch()

    h_state = agent.encode_state(batch['observations'])
    h_rgb = agent.encode_rgb(batch['third_person_observations'])

    assert h_state.shape == (batch['observations'].shape[0], agent.config['shared_latent_dim'])
    assert h_rgb.shape == h_state.shape
    np.testing.assert_allclose(np.asarray(jnp.linalg.norm(h_state, axis=-1)), 1.0, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(jnp.linalg.norm(h_rgb, axis=-1)), 1.0, rtol=1e-5, atol=1e-5)


def test_pretrain_update_logs_tcn_alignment_metrics():
    agent, batch = _make_agent_and_batch()

    agent, info = agent.pretrain(batch)

    for key in [
        'infonce/loss',
        'infonce/current_retrieval_acc_state_to_rgb',
        'bc_rgb/bc_loss',
        'bc_state/bc_loss',
        'flow_occupancy/neg_elbo_loss',
        'actor_modality/action_mse_rgb_state',
    ]:
        assert key in info
        assert bool(jnp.isfinite(info[key]))


def test_state_only_action_sampling_shape_and_range():
    agent, batch = _make_agent_and_batch()

    actions = agent.sample_actions(
        batch['observations'],
        seed=jax.random.PRNGKey(0),
        temperature=1.0,
    )

    assert actions.shape == batch['actions'].shape
    assert bool(jnp.all(actions <= 1.0))
    assert bool(jnp.all(actions >= -1.0))


def test_finetune_freezes_state_and_rgb_encoder_params():
    agent, batch = _make_agent_and_batch()
    agent, _ = agent.pretrain(batch)
    rgb_params_before = jax.tree_util.tree_map(
        lambda x: x.copy(),
        agent.network.params['modules_rgb_encoder'],
    )
    state_params_before = jax.tree_util.tree_map(
        lambda x: x.copy(),
        agent.network.params['modules_state_encoder'],
    )
    actor_params_before = jax.tree_util.tree_map(
        lambda x: x.copy(),
        agent.network.params['modules_actor'],
    )

    agent, info = agent.finetune(batch, full_update=True)

    assert _tree_max_abs_delta(rgb_params_before, agent.network.params['modules_rgb_encoder']) == 0.0
    assert _tree_max_abs_delta(state_params_before, agent.network.params['modules_state_encoder']) == 0.0
    assert _tree_max_abs_delta(actor_params_before, agent.network.params['modules_actor']) > 0.0
    assert bool(jnp.isfinite(info['reward/reward_loss']))
    assert bool(jnp.isfinite(info['loss/total_loss']))


if __name__ == '__main__':
    test_symmetric_infonce_prefers_matched_pairs()
    test_state_rgb_encoders_emit_unit_shared_latents()
    test_pretrain_update_logs_tcn_alignment_metrics()
    test_state_only_action_sampling_shape_and_range()
    test_finetune_freezes_state_and_rgb_encoder_params()
    print('cross-modal TCN tests passed')
