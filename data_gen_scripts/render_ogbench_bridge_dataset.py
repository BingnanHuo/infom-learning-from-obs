"""Render third-person RGB frames for an existing OGBench state dataset.

This path preserves the source dataset's ego/state observations, actions,
terminals, qpos, and qvel arrays. It only adds synchronized third-person RGB
renders plus episode metadata needed by the bridge loader.
"""

from __future__ import annotations

import os

import gymnasium
import numpy as np
import ogbench.manipspace  # noqa
from absl import app, flags
from tqdm import trange

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "cube-single-v0", "State-control OGBench environment name.")
flags.DEFINE_string("source_path", None, "Existing OGBench .npz file to augment.")
flags.DEFINE_string("save_path", None, "Output bridge .npz path.")
flags.DEFINE_string("render_camera", "front_pixels", "Camera used for the auxiliary third-person view.")
flags.DEFINE_integer("render_width", 64, "Rendered auxiliary view width.")
flags.DEFINE_integer("render_height", 64, "Rendered auxiliary view height.")
flags.DEFINE_integer("seed", 0, "Environment/render initialization seed.")
flags.DEFINE_integer("max_episodes", None, "Optional complete-episode subset for smoke tests.")
flags.DEFINE_bool("overwrite", False, "Whether to overwrite an existing output file.")
flags.DEFINE_bool("check_observations", True, "Whether to compare restored env observations against source samples.")
flags.DEFINE_integer("check_count", 16, "Number of source observations to spot-check.")


def normalize_npz_path(path):
    path = os.path.expanduser(path)
    _, ext = os.path.splitext(path)
    if ext == "":
        return f"{path}.npz"
    if ext != ".npz":
        raise ValueError(f"Expected an .npz path, got: {path}")
    return path


def complete_episode_stop(terminals, max_episodes):
    if max_episodes is None:
        return len(terminals)
    if max_episodes <= 0:
        raise ValueError("max_episodes must be positive when set.")
    terminal_idxs = np.flatnonzero(terminals)
    if len(terminal_idxs) < max_episodes:
        raise ValueError(f"Requested {max_episodes} episodes, but source only has {len(terminal_idxs)}.")
    return int(terminal_idxs[max_episodes - 1] + 1)


def build_episode_metadata(terminals):
    episode_ids = np.empty(len(terminals), dtype=np.int32)
    timesteps = np.empty(len(terminals), dtype=np.int32)
    episode_id = 0
    timestep = 0
    for idx, terminal in enumerate(terminals):
        episode_ids[idx] = episode_id
        timesteps[idx] = timestep
        if terminal:
            episode_id += 1
            timestep = 0
        else:
            timestep += 1
    return episode_ids, timesteps


def sample_check_indices(size, count):
    if size == 0 or count <= 0:
        return set()
    count = min(size, count)
    return set(np.unique(np.linspace(0, size - 1, num=count, dtype=np.int64)).tolist())


def main(_):
    if FLAGS.source_path is None:
        raise ValueError("--source_path is required.")
    if FLAGS.save_path is None:
        raise ValueError("--save_path is required.")

    source_path = normalize_npz_path(FLAGS.source_path)
    save_path = normalize_npz_path(FLAGS.save_path)
    if os.path.exists(save_path) and not FLAGS.overwrite:
        raise FileExistsError(f"Output already exists: {save_path}. Pass --overwrite to replace it.")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    with np.load(source_path) as source_file:
        required_keys = ["observations", "actions", "terminals", "qpos", "qvel"]
        missing = [key for key in required_keys if key not in source_file]
        if missing:
            raise KeyError(f"Missing required source keys in {source_path}: {missing}")

        terminals = source_file["terminals"][...].astype(bool, copy=False)
        stop = complete_episode_stop(terminals, FLAGS.max_episodes)
        dataset = {key: source_file[key][...][:stop] for key in source_file.files}

    terminals = dataset["terminals"].astype(bool, copy=False)
    episode_ids, timesteps = build_episode_metadata(terminals)
    dataset["episode_ids"] = episode_ids
    dataset["timesteps"] = timesteps

    env = gymnasium.make(
        FLAGS.env_name,
        terminate_at_goal=False,
        mode="data_collection",
        max_episode_steps=1001,
        width=FLAGS.render_width,
        height=FLAGS.render_height,
        visualize_info=False,
    )
    env.reset(seed=FLAGS.seed)
    env.action_space.seed(FLAGS.seed)

    num_transitions = len(terminals)
    third_person_observations = np.empty(
        (num_transitions, FLAGS.render_height, FLAGS.render_width, 3),
        dtype=np.uint8,
    )
    check_idxs = sample_check_indices(num_transitions, FLAGS.check_count if FLAGS.check_observations else 0)
    max_observation_diff = 0.0

    for idx in trange(num_transitions):
        env.unwrapped.set_state(dataset["qpos"][idx], dataset["qvel"][idx])
        if idx in check_idxs:
            restored_observation = np.asarray(env.unwrapped.compute_observation(), dtype=dataset["observations"].dtype)
            diff = np.max(np.abs(restored_observation - dataset["observations"][idx]))
            max_observation_diff = max(max_observation_diff, float(diff))
        third_person_observations[idx] = env.unwrapped.render(camera=FLAGS.render_camera)

    if FLAGS.check_observations:
        print(f"Max restored observation diff over {len(check_idxs)} checks: {max_observation_diff}")
        if max_observation_diff > 1e-4:
            raise ValueError(
                "Restored qpos/qvel observations do not match source observations; "
                f"max diff {max_observation_diff}."
            )

    dataset["third_person_observations"] = third_person_observations
    np.savez_compressed(save_path, **dataset)
    print(f"Saved bridge dataset to {save_path}")


if __name__ == "__main__":
    app.run(main)
