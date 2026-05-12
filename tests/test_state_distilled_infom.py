import os
import sys
import tempfile

import numpy as np
import jax
import jax.numpy as jnp

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agents.cross_modal_state_distilled_infom import CrossModalStateDistilledInFOMAgent, get_config
from envs.env_utils import _truncate_dataset
from envs.ogbench_bridge_utils import load_dataset
from utils.datasets import DeviceBridgeDataset, Dataset


def _state_stats(observations, state_clip=5.0, eps=1e-8):
    observations = observations.astype(np.float32)
    mean = observations.mean(axis=0)
    std = np.sqrt(observations.var(axis=0) + eps)
    latents = np.clip((observations - mean) / std, -state_clip, state_clip)
    return dict(
        state_mean=mean,
        state_std=std,
        latent_min=latents.min(axis=0),
        latent_max=latents.max(axis=0),
    )


def _tiny_config():
    config = get_config()
    config.lr = 1e-3
    config.batch_size = 4
    config.rgb_encoder = 'impala_debug'
    config.rgb_frame_stack = 3
    config.rgb_p_aug = None
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
    rng = np.random.default_rng(0)
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
    stats = _state_stats(dataset['observations'], state_clip=config.state_clip, eps=config.state_norm_eps)
    agent = CrossModalStateDistilledInFOMAgent.create(
        seed=0,
        ex_observations=batch['observations'],
        ex_actions=batch['actions'],
        config=config,
        ex_third_person_observations=batch['third_person_observations'],
        **stats,
    )
    return agent, batch


def _tree_max_abs_delta(a, b):
    leaves = jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda x, y: jnp.max(jnp.abs(x - y)), a, b))
    return float(jnp.max(jnp.stack(leaves)))


def test_dataset_aux_rgb_frame_stack_and_next_actions():
    dataset = _synthetic_dataset()
    batch = dataset.sample(3, idxs=np.array([0, 1, 6]))

    assert batch['observations'].shape == (3, 5)
    assert batch['third_person_observations'].shape == (3, 16, 16, 9)
    assert batch['next_third_person_observations'].shape == (3, 16, 16, 9)
    np.testing.assert_allclose(batch['next_actions'], dataset['actions'][np.array([1, 2, 7])])
    np.testing.assert_array_equal(
        batch['third_person_observations'][0, :, :, :3],
        batch['third_person_observations'][0, :, :, 3:6],
    )


def test_bridge_loader_can_skip_third_person_rgb():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'bridge-cube-single-play-v0.npz')
        num_steps = 6
        terminals = np.zeros((num_steps,), dtype=np.float32)
        terminals[[2, 5]] = 1.0
        np.savez(
            path,
            observations=np.arange(num_steps * 5, dtype=np.float32).reshape(num_steps, 5),
            third_person_observations=np.zeros((num_steps, 8, 8, 3), dtype=np.uint8),
            actions=np.ones((num_steps, 2), dtype=np.float32),
            terminals=terminals,
            qpos=np.zeros((num_steps, 3), dtype=np.float32),
            qvel=np.zeros((num_steps, 3), dtype=np.float32),
            episode_ids=np.repeat(np.arange(2, dtype=np.int32), 3),
            timesteps=np.tile(np.arange(3, dtype=np.int32), 2),
        )

        state_only = load_dataset(path, include_third_person=False)
        paired = load_dataset(path, include_third_person=True)

    assert 'third_person_observations' not in state_only
    assert 'next_third_person_observations' not in state_only
    assert 'third_person_observations' in paired
    assert 'next_third_person_observations' in paired
    assert state_only['observations'].shape == paired['observations'].shape


def test_truncate_dataset_makes_compact_copies():
    source = {
        'observations': np.arange(20, dtype=np.float32).reshape(10, 2),
        'actions': np.arange(30, dtype=np.float32).reshape(10, 3),
    }

    truncated = _truncate_dataset(source, 4)

    assert truncated['observations'].shape == (4, 2)
    assert truncated['actions'].shape == (4, 3)
    assert truncated['observations'].flags.c_contiguous
    assert not np.shares_memory(truncated['observations'], source['observations'])
    assert not np.shares_memory(truncated['actions'], source['actions'])


def test_device_bridge_dataset_matches_cpu_fixed_indices():
    dataset = _synthetic_dataset()
    idxs = np.array([0, 1, 6, 7], dtype=np.int32)
    cpu_batch = dataset.sample(4, idxs=idxs)
    device_dataset = DeviceBridgeDataset.create_from_dataset(
        dataset,
        seed=0,
        rgb_frame_stack=3,
        p_aug=None,
    )
    device_batch = jax.tree_util.tree_map(np.asarray, device_dataset.sample(4, idxs=idxs))

    assert device_dataset.nbytes() > 0
    for key in [
        'observations',
        'next_observations',
        'actions',
        'next_actions',
        'terminals',
        'third_person_observations',
        'next_third_person_observations',
    ]:
        np.testing.assert_array_equal(device_batch[key], cpu_batch[key])


def test_state_normalizer_and_state_only_action_sampling():
    agent, batch = _make_agent_and_batch()

    h_state = agent.encode_state(batch['observations'])
    assert h_state.shape == batch['observations'].shape
    assert bool(jnp.all(jnp.isfinite(h_state)))

    actions = agent.sample_actions(
        batch['observations'],
        seed=jax.random.PRNGKey(0),
        temperature=1.0,
    )
    assert actions.shape == batch['actions'].shape
    assert bool(jnp.all(actions <= 1.0))
    assert bool(jnp.all(actions >= -1.0))


def test_pretrain_update_logs_cross_modal_metrics():
    agent, batch = _make_agent_and_batch()

    agent, info = agent.pretrain(batch)

    for key in [
        'align/align_loss',
        'bc_rgb/bc_loss',
        'bc_state/bc_loss',
        'flow_occupancy/neg_elbo_loss',
        'actor_modality/action_mse_rgb_state',
    ]:
        assert key in info
        assert bool(jnp.isfinite(info[key]))


def test_finetune_freezes_rgb_encoder_params():
    agent, batch = _make_agent_and_batch()
    agent, _ = agent.pretrain(batch)
    rgb_params_before = jax.tree_util.tree_map(
        lambda x: x.copy(),
        agent.network.params['modules_rgb_encoder'],
    )

    agent, info = agent.finetune(batch, full_update=True)
    rgb_params_after = agent.network.params['modules_rgb_encoder']

    assert _tree_max_abs_delta(rgb_params_before, rgb_params_after) == 0.0
    assert bool(jnp.isfinite(info['reward/reward_loss']))
    assert bool(jnp.isfinite(info['loss/total_loss']))


if __name__ == '__main__':
    test_dataset_aux_rgb_frame_stack_and_next_actions()
    test_bridge_loader_can_skip_third_person_rgb()
    test_truncate_dataset_makes_compact_copies()
    test_device_bridge_dataset_matches_cpu_fixed_indices()
    test_state_normalizer_and_state_only_action_sampling()
    test_pretrain_update_logs_cross_modal_metrics()
    test_finetune_freezes_rgb_encoder_params()
    print('state-distilled tests passed')
