#!/usr/bin/env python3
"""Render evaluation rollouts from an OGBench bridge checkpoint."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
from pathlib import Path

import flax
import jax.numpy as jnp
import ml_collections
import numpy as np

from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset
from utils.evaluation import evaluate
from utils.flax_utils import find_latest_training_checkpoint, restore_training_checkpoint


def load_config(path: Path) -> ml_collections.ConfigDict:
    spec = importlib.util.spec_from_file_location("agent_config", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load agent config from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()


def compute_state_stats(dataset, state_clip: float, epsilon: float) -> dict:
    observations = np.asarray(dataset["observations"], dtype=np.float32)
    state_mean = np.mean(observations, axis=0)
    state_std = np.sqrt(np.var(observations, axis=0) + epsilon)
    state_latents = np.clip((observations - state_mean) / state_std, -state_clip, state_clip)
    return {
        "state_mean": state_mean.astype(np.float32),
        "state_std": state_std.astype(np.float32),
        "latent_min": np.min(state_latents, axis=0).astype(np.float32),
        "latent_max": np.max(state_latents, axis=0).astype(np.float32),
    }


def restore_any_checkpoint(agent, path: Path):
    if path.is_file() and path.name.startswith("params_"):
        with path.open("rb") as f:
            load_dict = pickle.load(f)
        agent = flax.serialization.from_state_dict(agent, load_dict["agent"])
        return agent, str(path)

    latest = find_latest_training_checkpoint(str(path))
    if latest is not None:
        agent, _, restored_path = restore_training_checkpoint(agent, str(path))
        return agent, restored_path

    raise FileNotFoundError(f"No params_*.pkl or full training checkpoint found at {path}")


def save_renders(renders, out_dir: Path, stem: str, fps: int) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        frames_path = out_dir / f"{stem}_frames.npz"
        np.savez_compressed(frames_path, **{f"episode_{i}": render for i, render in enumerate(renders)})
        return [str(frames_path), f"imageio unavailable: {exc}"]

    for i, render in enumerate(renders):
        if render.size == 0:
            continue
        mp4_path = out_dir / f"{stem}_episode{i}.mp4"
        gif_path = out_dir / f"{stem}_episode{i}.gif"
        try:
            imageio.mimsave(mp4_path, list(render), fps=fps)
            saved.append(str(mp4_path))
        except Exception as exc:
            saved.append(f"mp4 failed for episode {i}: {exc}")
        try:
            imageio.mimsave(gif_path, list(render), fps=fps)
            saved.append(str(gif_path))
        except Exception as exc:
            saved.append(f"gif failed for episode {i}: {exc}")
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--agent", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shared-latent-dim", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--video-episodes", type=int, default=1)
    parser.add_argument("--video-frame-skip", type=int, default=3)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--pretraining-size", type=int, default=1_000_000)
    parser.add_argument("--finetuning-size", type=int, default=500_000)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    config = load_config(args.agent)
    if args.shared_latent_dim is not None:
        config.shared_latent_dim = args.shared_latent_dim
    cross_modal = config["agent_name"] in {
        "cross_modal_state_distilled_infom",
        "cross_modal_tcn_infom",
    }
    if not cross_modal:
        raise ValueError(f"Demo script is intended for cross-modal bridge agents, got {config['agent_name']}")

    _, _, pretraining_train_dataset, _ = make_env_and_datasets(
        args.env_name,
        frame_stack=None,
        max_size=args.pretraining_size,
        reward_free=True,
        dataset_dir=str(args.dataset_dir),
        include_third_person=True,
    )
    _, eval_env, finetuning_train_dataset, _ = make_env_and_datasets(
        args.env_name,
        frame_stack=None,
        max_size=args.finetuning_size,
        reward_free=False,
        dataset_dir=str(args.dataset_dir),
        include_third_person=True,
    )

    pretraining_train_dataset = Dataset.create(**pretraining_train_dataset)
    finetuning_train_dataset = Dataset.create(**finetuning_train_dataset)

    for dataset in [pretraining_train_dataset, finetuning_train_dataset]:
        dataset.obs_norm_type = "none"
        dataset.p_aug = config["rgb_p_aug"]
        dataset.frame_stack = None
        dataset.aux_frame_stack = config["rgb_frame_stack"]
        dataset.aux_frame_stack_keys = ("third_person_observations",)
        dataset.aug_keys = ("third_person_observations", "next_third_person_observations")
        dataset.return_next_actions = True
        dataset.normalize_observations()

    create_kwargs = compute_state_stats(
        pretraining_train_dataset,
        state_clip=config["state_clip"],
        epsilon=config["state_norm_eps"],
    )
    if config["agent_name"] == "cross_modal_tcn_infom":
        latent_dim = config["shared_latent_dim"]
        create_kwargs["latent_min"] = -np.ones(latent_dim, dtype=np.float32)
        create_kwargs["latent_max"] = np.ones(latent_dim, dtype=np.float32)

    example_batch = pretraining_train_dataset.sample(1)
    create_kwargs["ex_third_person_observations"] = example_batch["third_person_observations"]
    agent = agents[config["agent_name"]].create(
        args.seed,
        example_batch["observations"],
        example_batch["actions"],
        config,
        **create_kwargs,
    )
    agent, restored_path = restore_any_checkpoint(agent, args.checkpoint)

    eval_info, trajs, renders = evaluate(
        agent=agent,
        env=eval_env,
        dataset=finetuning_train_dataset,
        num_eval_episodes=args.eval_episodes,
        num_video_episodes=args.video_episodes,
        video_frame_skip=args.video_frame_skip,
    )

    stem = f"{config['agent_name']}_{args.env_name}_seed{args.seed}"
    saved = save_renders(renders, args.out_dir, stem, args.fps)
    summary = {
        "env_name": args.env_name,
        "agent_name": config["agent_name"],
        "checkpoint": str(args.checkpoint),
        "restored_path": str(restored_path),
        "seed": args.seed,
        "eval_info": {k: float(v) for k, v in eval_info.items()},
        "num_eval_episodes": args.eval_episodes,
        "num_video_episodes": args.video_episodes,
        "saved_artifacts": saved,
        "trajectory_lengths": [len(traj["reward"]) for traj in trajs],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
