from collections import defaultdict
import os
import re
import sys

import gymnasium
import numpy as np
import ogbench.manipspace  # noqa
from absl import app, flags
from ogbench.manipspace.oracles.markov.button_markov import ButtonMarkovOracle
from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle
from ogbench.manipspace.oracles.markov.drawer_markov import DrawerMarkovOracle
from ogbench.manipspace.oracles.markov.window_markov import WindowMarkovOracle
from ogbench.manipspace.oracles.plan.button_plan import ButtonPlanOracle
from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle
from ogbench.manipspace.oracles.plan.drawer_plan import DrawerPlanOracle
from ogbench.manipspace.oracles.plan.window_plan import WindowPlanOracle
from ogbench.utils import DEFAULT_DATASET_DIR
from tqdm import trange

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from envs.bridge_wrappers import ThirdPersonRenderWrapper

FLAGS = flags.FLAGS
DATASET_TYPE_TOKENS = {'play', 'noisy'}

flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-single-v0', 'State-control OGBench environment name.')
flags.DEFINE_string('dataset_type', 'play', 'Dataset type.')
flags.DEFINE_string('dataset_dir', DEFAULT_DATASET_DIR, 'Directory to save generated datasets.')
flags.DEFINE_string('save_path', None, 'Optional explicit train dataset path.')
flags.DEFINE_string('render_camera', 'front_pixels', 'Camera used for the auxiliary third-person view.')
flags.DEFINE_integer('render_width', 64, 'Rendered auxiliary view width.')
flags.DEFINE_integer('render_height', 64, 'Rendered auxiliary view height.')
flags.DEFINE_float('noise', 0.1, 'Action noise level.')
flags.DEFINE_float('noise_smoothing', 0.5, 'Action noise smoothing level for PlanOracle.')
flags.DEFINE_float('min_norm', 0.4, 'Minimum action norm for MarkovOracle.')
flags.DEFINE_float('p_random_action', 0.0, 'Probability of selecting a random action.')
flags.DEFINE_integer('num_episodes', 1000, 'Number of training episodes.')
flags.DEFINE_integer('max_episode_steps', 1001, 'Maximum episode length.')


def default_save_path():
    dataset_dir = os.path.expanduser(FLAGS.dataset_dir)
    os.makedirs(dataset_dir, exist_ok=True)
    version_match = re.match(r'^(?P<stem>.+)-(?P<version>v\d+)$', FLAGS.env_name)
    if version_match is None:
        dataset_stem, dataset_version = FLAGS.env_name, 'v0'
    else:
        dataset_stem = version_match.group('stem')
        dataset_version = version_match.group('version')
    dataset_stem_parts = dataset_stem.split('-')
    if 'singletask' in dataset_stem_parts:
        # Singletask names are virtual loader names; the physical file is the base dataset.
        singletask_pos = dataset_stem_parts.index('singletask')
        if singletask_pos > 0 and dataset_stem_parts[singletask_pos - 1] in DATASET_TYPE_TOKENS:
            dataset_stem_parts[singletask_pos - 1] = FLAGS.dataset_type
            dataset_stem_parts = dataset_stem_parts[:singletask_pos]
        else:
            dataset_stem_parts = dataset_stem_parts[:singletask_pos] + [FLAGS.dataset_type]
        dataset_stem = '-'.join(dataset_stem_parts)
    else:
        dataset_stem = f'{dataset_stem}-{FLAGS.dataset_type}'
    return os.path.join(
        dataset_dir,
        f'bridge-{dataset_stem}-{dataset_version}.npz',
    )


def normalize_npz_path(path):
    path = os.path.expanduser(path)
    _, ext = os.path.splitext(path)
    if ext == '':
        return f'{path}.npz'
    if ext != '.npz':
        raise ValueError(f'Expected an .npz output path, got: {path}')
    return path


def resolve_output_paths():
    train_path = normalize_npz_path(FLAGS.save_path) if FLAGS.save_path else default_save_path()
    train_root, _ = os.path.splitext(train_path)
    val_path = f'{train_root}-val.npz'
    output_dir = os.path.dirname(train_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    return train_path, val_path


def reset_seed_for_attempt(ep_idx, retry_idx, num_episodes):
    return FLAGS.seed + ep_idx + retry_idx * num_episodes


def build_oracles(env):
    oracle_type = 'plan' if FLAGS.dataset_type == 'play' else 'markov'
    if 'cube' in FLAGS.env_name:
        if oracle_type == 'markov':
            return {
                'cube': CubeMarkovOracle(env=env, min_norm=FLAGS.min_norm),
            }
        return {
            'cube': CubePlanOracle(env=env, noise=FLAGS.noise, noise_smoothing=FLAGS.noise_smoothing),
        }
    if 'scene' in FLAGS.env_name:
        if oracle_type == 'markov':
            return {
                'cube': CubeMarkovOracle(env=env, min_norm=FLAGS.min_norm, max_step=100),
                'button': ButtonMarkovOracle(env=env, min_norm=FLAGS.min_norm),
                'drawer': DrawerMarkovOracle(env=env, min_norm=FLAGS.min_norm),
                'window': WindowMarkovOracle(env=env, min_norm=FLAGS.min_norm),
            }
        return {
            'cube': CubePlanOracle(env=env, noise=FLAGS.noise, noise_smoothing=FLAGS.noise_smoothing),
            'button': ButtonPlanOracle(env=env, noise=FLAGS.noise, noise_smoothing=FLAGS.noise_smoothing),
            'drawer': DrawerPlanOracle(env=env, noise=FLAGS.noise, noise_smoothing=FLAGS.noise_smoothing),
            'window': WindowPlanOracle(env=env, noise=FLAGS.noise, noise_smoothing=FLAGS.noise_smoothing),
        }
    if 'puzzle' in FLAGS.env_name:
        if oracle_type == 'markov':
            return {
                'button': ButtonMarkovOracle(env=env, min_norm=FLAGS.min_norm, gripper_always_closed=True),
            }
        return {
            'button': ButtonPlanOracle(
                env=env,
                noise=FLAGS.noise,
                noise_smoothing=FLAGS.noise_smoothing,
                gripper_always_closed=True,
            ),
        }
    raise ValueError(f'Unsupported OGBench bridge environment: {FLAGS.env_name}')


def append_transition(
    dataset,
    observation,
    third_person_observation,
    action,
    info,
    done,
    episode_id,
    timestep,
    has_button_states=False,
):
    dataset['observations'].append(observation)
    dataset['third_person_observations'].append(third_person_observation)
    dataset['actions'].append(action)
    dataset['terminals'].append(done)
    dataset['qpos'].append(info['prev_qpos'])
    dataset['qvel'].append(info['prev_qvel'])
    if has_button_states:
        dataset['button_states'].append(info['prev_button_states'])
    dataset['episode_ids'].append(episode_id)
    dataset['timesteps'].append(timestep)


def finalize_split(dataset, stop_idx):
    split_dataset = {}
    dtypes = {
        'observations': np.float32,
        'third_person_observations': np.uint8,
        'actions': np.float32,
        'terminals': bool,
        'qpos': np.float32,
        'qvel': np.float32,
        'button_states': np.int64,
        'episode_ids': np.int32,
        'timesteps': np.int32,
    }
    for key, values in dataset.items():
        split_dataset[key] = np.array(values[:stop_idx], dtype=dtypes[key])
    return split_dataset


def main(_):
    assert FLAGS.dataset_type in ['play', 'noisy']
    np.random.seed(FLAGS.seed)

    base_env = gymnasium.make(
        FLAGS.env_name,
        terminate_at_goal=False,
        mode='data_collection',
        max_episode_steps=FLAGS.max_episode_steps,
        width=FLAGS.render_width,
        height=FLAGS.render_height,
        visualize_info=False,
    )
    env = ThirdPersonRenderWrapper(base_env, camera=FLAGS.render_camera)
    oracles = build_oracles(env)
    has_button_states = hasattr(env.unwrapped, '_cur_button_states')

    dataset = defaultdict(list)
    total_steps = 0
    total_train_steps = 0
    num_train_episodes = FLAGS.num_episodes
    num_val_episodes = max(1, FLAGS.num_episodes // 10)

    total_episodes = num_train_episodes + num_val_episodes

    for ep_idx in trange(total_episodes):
        retry_idx = 0
        while True:
            observation, info = env.reset(seed=reset_seed_for_attempt(ep_idx, retry_idx, total_episodes))
            third_person_observation = info['third_person_observation']

            if 'single' in FLAGS.env_name:
                p_stack = 0.0
            elif 'double' in FLAGS.env_name:
                p_stack = np.random.uniform(0.0, 0.25)
            elif 'triple' in FLAGS.env_name:
                p_stack = np.random.uniform(0.05, 0.35)
            elif 'quadruple' in FLAGS.env_name:
                p_stack = np.random.uniform(0.1, 0.5)
            else:
                p_stack = 0.5

            if FLAGS.dataset_type == 'noisy':
                action_noise = np.random.uniform(0.0, FLAGS.noise)

            agent = oracles[info['privileged/target_task']]
            agent.reset(observation, info)

            done = False
            timestep = 0
            ep_qpos = []

            while not done:
                if np.random.rand() < FLAGS.p_random_action:
                    action = env.action_space.sample()
                else:
                    action = np.array(agent.select_action(observation, info))
                    if FLAGS.dataset_type == 'noisy':
                        action = action + np.random.normal(
                            0,
                            [action_noise, action_noise, action_noise, action_noise * 3, action_noise * 10],
                            action.shape,
                        )
                action = np.clip(action, -1, 1)

                next_observation, _, terminated, truncated, next_info = env.step(action)
                done = terminated or truncated

                append_transition(
                    dataset=dataset,
                    observation=observation,
                    third_person_observation=third_person_observation,
                    action=action,
                    info=next_info,
                    done=done,
                    episode_id=ep_idx,
                    timestep=timestep,
                    has_button_states=has_button_states,
                )
                ep_qpos.append(next_info['prev_qpos'])

                if agent.done:
                    agent_observation, agent_info = env.unwrapped.set_new_target(p_stack=p_stack)
                    agent_info = dict(agent_info)
                    agent_info['third_person_observation'] = env.unwrapped.render(camera=FLAGS.render_camera)
                    agent = oracles[agent_info['privileged/target_task']]
                    agent.reset(agent_observation, agent_info)

                observation = next_observation
                info = next_info
                third_person_observation = next_info['third_person_observation']
                timestep += 1

            if 'scene' in FLAGS.env_name:
                is_healthy = True
                ep_qpos = np.array(ep_qpos)
                block_xyzs = ep_qpos[:, 14:17]
                if (block_xyzs[:, 1] >= 0.29).any():
                    is_healthy = False
                if ((block_xyzs[:, 1] <= -0.3) & ((block_xyzs[:, 2] < 0.06) | (block_xyzs[:, 2] > 0.08))).any():
                    is_healthy = False

                if is_healthy:
                    break

                print('Unhealthy episode, retrying...', flush=True)
                for key in dataset.keys():
                    dataset[key] = dataset[key][:-timestep]
                retry_idx += 1
            else:
                break

        total_steps += timestep
        if ep_idx < num_train_episodes:
            total_train_steps += timestep

    print('Total steps:', total_steps)

    train_path, val_path = resolve_output_paths()

    train_dataset = finalize_split(dataset, total_train_steps)
    val_dataset = {}
    for key, values in dataset.items():
        dtype = train_dataset[key].dtype
        val_dataset[key] = np.array(values[total_train_steps:], dtype=dtype)

    np.savez_compressed(train_path, **train_dataset)
    np.savez_compressed(val_path, **val_dataset)


if __name__ == '__main__':
    app.run(main)
