import os

import gymnasium
import numpy as np

from ogbench.relabel_utils import add_oracle_reps, relabel_dataset
from ogbench.utils import DEFAULT_DATASET_DIR

from envs.bridge_wrappers import ThirdPersonRenderWrapper

BRIDGE_DATASET_PREFIX = 'bridge-'


def load_dataset(
    dataset_path,
    ob_dtype=np.float32,
    action_dtype=np.float32,
    compact_dataset=False,
):
    """Load a paired OGBench bridge dataset while preserving auxiliary fields."""
    with np.load(dataset_path) as file:
        required_keys = [
            'observations',
            'third_person_observations',
            'actions',
            'terminals',
            'qpos',
            'qvel',
            'episode_ids',
            'timesteps',
        ]
        missing_keys = [k for k in required_keys if k not in file]
        if missing_keys:
            raise KeyError(f'Missing keys in bridge dataset {dataset_path}: {missing_keys}')

        dataset = dict(
            observations=file['observations'][...].astype(ob_dtype, copy=False),
            third_person_observations=file['third_person_observations'][...].astype(np.uint8, copy=False),
            actions=file['actions'][...].astype(action_dtype, copy=False),
            terminals=file['terminals'][...].astype(np.float32, copy=False),
            qpos=file['qpos'][...].astype(np.float32, copy=False),
            qvel=file['qvel'][...].astype(np.float32, copy=False),
            episode_ids=file['episode_ids'][...].astype(np.int32, copy=False),
            timesteps=file['timesteps'][...].astype(np.int32, copy=False),
        )
        if 'button_states' in file:
            dataset['button_states'] = file['button_states'][...].astype(np.int64, copy=False)

    if compact_dataset:
        original_terminals = dataset['terminals']
        new_terminals = np.concatenate([original_terminals[1:], [1.0]])
        dataset['valids'] = 1.0 - original_terminals
        dataset['terminals'] = np.minimum(original_terminals + new_terminals, 1.0).astype(np.float32)
        return dataset

    original_terminals = dataset['terminals']
    ob_mask = (1.0 - original_terminals).astype(bool)
    next_ob_mask = np.concatenate([[False], ob_mask[:-1]])

    dataset['next_observations'] = dataset['observations'][next_ob_mask]
    dataset['next_third_person_observations'] = dataset['third_person_observations'][next_ob_mask]

    dataset['observations'] = dataset['observations'][ob_mask]
    dataset['third_person_observations'] = dataset['third_person_observations'][ob_mask]
    dataset['actions'] = dataset['actions'][ob_mask]
    dataset['qpos'] = dataset['qpos'][ob_mask]
    dataset['qvel'] = dataset['qvel'][ob_mask]
    if 'button_states' in dataset:
        dataset['button_states'] = dataset['button_states'][ob_mask]
    dataset['episode_ids'] = dataset['episode_ids'][ob_mask]
    dataset['timesteps'] = dataset['timesteps'][ob_mask]

    new_terminals = np.concatenate([original_terminals[1:], [1.0]])
    dataset['terminals'] = new_terminals[ob_mask].astype(np.float32)

    return dataset


def make_env_and_datasets(
    dataset_name,
    dataset_dir=DEFAULT_DATASET_DIR,
    compact_dataset=False,
    env_only=False,
    **env_kwargs,
):
    """Make a state-control OGBench env and load a paired bridge dataset."""
    if not dataset_name.startswith(BRIDGE_DATASET_PREFIX):
        raise ValueError(
            f'Bridge dataset names must start with {BRIDGE_DATASET_PREFIX!r}: {dataset_name}'
        )

    base_dataset_name = dataset_name[len(BRIDGE_DATASET_PREFIX):]
    splits = base_dataset_name.split('-')
    relabel_rewards = False
    use_oracle_rep = False

    if 'singletask' in splits:
        pos = splits.index('singletask')
        if 'ft' in splits:
            env_name = '-'.join(splits[: pos - 2] + splits[pos:])
        else:
            env_name = '-'.join(splits[: pos - 1] + splits[pos:])
        base_dataset_name = '-'.join(splits[:pos] + splits[-1:])
        relabel_rewards = True
    elif 'oraclerep' in splits:
        env_name = '-'.join(splits[:-3] + splits[-1:])
        base_dataset_name = '-'.join(splits[:-2] + splits[-1:])
        use_oracle_rep = True
    else:
        env_name = '-'.join(splits[:-2] + splits[-1:])

    if use_oracle_rep:
        env = gymnasium.make(env_name, use_oracle_rep=True, **env_kwargs)
    else:
        env = gymnasium.make(env_name, **env_kwargs)

    if env_only:
        return env

    dataset_dir = os.path.expanduser(dataset_dir)
    train_dataset_path = os.path.join(dataset_dir, f'{BRIDGE_DATASET_PREFIX}{base_dataset_name}.npz')
    val_dataset_path = os.path.join(dataset_dir, f'{BRIDGE_DATASET_PREFIX}{base_dataset_name}-val.npz')

    if not os.path.exists(train_dataset_path) or not os.path.exists(val_dataset_path):
        raise FileNotFoundError(
            'Bridge dataset files not found. Expected '
            f'{train_dataset_path} and {val_dataset_path}.'
        )

    train_dataset = load_dataset(train_dataset_path, compact_dataset=compact_dataset)
    val_dataset = load_dataset(val_dataset_path, compact_dataset=compact_dataset)

    if relabel_rewards:
        if ('scene' in env_name or 'puzzle' in env_name) and 'button_states' not in train_dataset:
            raise KeyError(
                'Bridge dataset is missing button_states, which are required to relabel '
                f'{env_name} single-task rewards.'
            )
        relabel_dataset(env_name, env, train_dataset)
        relabel_dataset(env_name, env, val_dataset)

    if use_oracle_rep:
        add_oracle_reps(env_name, env, train_dataset)
        add_oracle_reps(env_name, env, val_dataset)

    return env, train_dataset, val_dataset


def make_online_env(
    env_name,
    dataset_camera='front_pixels',
    width=64,
    height=64,
    info_key='third_person_observation',
    **env_kwargs,
):
    """Create a state-control env that also emits a synchronized third-person render."""
    env = gymnasium.make(
        env_name,
        width=width,
        height=height,
        visualize_info=False,
        **env_kwargs,
    )
    return ThirdPersonRenderWrapper(env, camera=dataset_camera, info_key=info_key)
