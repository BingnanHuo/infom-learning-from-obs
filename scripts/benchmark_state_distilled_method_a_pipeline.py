#!/usr/bin/env python3
"""Benchmark State-Distilled InFOM data and update costs."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import time
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import numpy as np

from agents.cross_modal_state_distilled_infom import (
    CrossModalStateDistilledInFOMAgent,
    get_config,
)
from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset


def current_rss_mb() -> float:
    try:
        with open("/proc/self/statm") as f:
            rss_pages = int(f.read().split()[1])
        return rss_pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except (OSError, IndexError, ValueError):
        return float("nan")


def tree_nbytes(tree: Any) -> int:
    if isinstance(tree, Mapping):
        return sum(tree_nbytes(value) for value in tree.values())
    if isinstance(tree, (list, tuple)):
        return sum(tree_nbytes(value) for value in tree)
    if hasattr(tree, "_dict"):
        return tree_nbytes(tree._dict)
    if hasattr(tree, "nbytes"):
        return int(tree.nbytes)

    total = 0
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "nbytes"):
            total += int(leaf.nbytes)
    return total


def block_tree(tree: Any) -> Any:
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return tree


def scalarize(tree: Any) -> Any:
    if isinstance(tree, dict):
        return {str(k): scalarize(v) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        return [scalarize(v) for v in tree]
    array = np.asarray(tree)
    if array.shape == ():
        value = array.item()
        if isinstance(value, (np.floating, float)):
            return float(value)
        if isinstance(value, (np.integer, int)):
            return int(value)
        if isinstance(value, (np.bool_, bool)):
            return bool(value)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "mean": float(np.mean(array)) if array.size else None,
    }


def compute_state_distillation_stats(dataset: Dataset, state_clip: float, epsilon: float) -> dict[str, np.ndarray]:
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


def configure_dataset(dataset: Dataset, config, p_aug: float | None) -> None:
    dataset.obs_norm_type = "none"
    dataset.p_aug = p_aug
    dataset.num_aug = 1
    dataset.inplace_aug = 1
    dataset.frame_stack = None
    dataset.aux_frame_stack = config["rgb_frame_stack"]
    dataset.aux_frame_stack_keys = ("third_person_observations",)
    dataset.aug_keys = ("third_person_observations", "next_third_person_observations")
    dataset.return_next_actions = True
    dataset.normalize_observations()


def parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-name", default="bridge-cube-single-play-singletask-task1-v0")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pretraining-size", type=int, default=250_000)
    parser.add_argument("--finetuning-size", type=int, default=125_000)
    parser.add_argument("--load-finetune", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sample-batches", type=int, default=50)
    parser.add_argument("--update-steps", type=int, default=20)
    parser.add_argument("--aug-probs", default="0.0,0.5,1.0")
    parser.add_argument("--rgb-frame-stack", type=int, default=3)
    parser.add_argument("--rgb-encoder", default="impala_small")
    parser.add_argument("--warmup-align-steps", type=int, default=10_000)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    config = get_config()
    config.batch_size = args.batch_size
    config.rgb_frame_stack = args.rgb_frame_stack
    config.rgb_encoder = args.rgb_encoder
    config.warmup_align_steps = args.warmup_align_steps
    config.expectile = 0.95
    config.kl_weight = 0.05
    config.alpha = 30

    results: dict[str, Any] = {
        "args": vars(args),
        "jax": {
            "default_backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
        },
        "timings": [],
        "rss_mb_start": current_rss_mb(),
    }

    @contextmanager
    def timed(name: str, extra: dict[str, Any] | None = None):
        gc.collect()
        start_rss = current_rss_mb()
        start = time.perf_counter()
        yield
        seconds = time.perf_counter() - start
        end_rss = current_rss_mb()
        record = {
            "name": name,
            "seconds": seconds,
            "rss_mb_before": start_rss,
            "rss_mb_after": end_rss,
            "rss_mb_delta": end_rss - start_rss,
        }
        if extra:
            record.update(extra)
        results["timings"].append(record)

    with timed("load_pretraining_datasets"):
        _, _, pretraining_train, pretraining_val = make_env_and_datasets(
            args.env_name,
            frame_stack=None,
            max_size=args.pretraining_size,
            reward_free=True,
            dataset_dir=args.dataset_dir,
            include_third_person=True,
        )
    results["pretraining_loaded"] = {
        "train_size": int(pretraining_train["observations"].shape[0]),
        "val_size": int(pretraining_val["observations"].shape[0]),
        "train_bytes": tree_nbytes(pretraining_train),
        "val_bytes": tree_nbytes(pretraining_val),
    }

    finetuning_train = None
    finetuning_val = None
    if args.load_finetune:
        with timed("load_finetuning_datasets"):
            _, _, finetuning_train, finetuning_val = make_env_and_datasets(
                args.env_name,
                frame_stack=None,
                max_size=args.finetuning_size,
                reward_free=False,
                dataset_dir=args.dataset_dir,
                include_third_person=True,
            )
        results["finetuning_loaded"] = {
            "train_size": int(finetuning_train["observations"].shape[0]),
            "val_size": int(finetuning_val["observations"].shape[0]),
            "train_bytes": tree_nbytes(finetuning_train),
            "val_bytes": tree_nbytes(finetuning_val),
        }

    with timed("wrap_datasets"):
        pretraining_train = Dataset.create(**pretraining_train)
        pretraining_val = Dataset.create(**pretraining_val)
        if finetuning_train is not None:
            finetuning_train = Dataset.create(**finetuning_train)
            finetuning_val = Dataset.create(**finetuning_val)

    with timed("compute_state_stats"):
        state_stats = compute_state_distillation_stats(
            pretraining_train,
            state_clip=config["state_clip"],
            epsilon=config["state_norm_eps"],
        )

    with timed("normalize_and_configure_datasets"):
        configure_dataset(pretraining_train, config, p_aug=0.0)
        configure_dataset(pretraining_val, config, p_aug=0.0)
        if finetuning_train is not None:
            configure_dataset(finetuning_train, config, p_aug=0.0)
            configure_dataset(finetuning_val, config, p_aug=0.0)

    sample_results = []
    for p_aug in parse_float_list(args.aug_probs):
        pretraining_train.p_aug = p_aug
        start = time.perf_counter()
        batch = None
        for _ in range(args.sample_batches):
            batch = pretraining_train.sample(args.batch_size)
        seconds = time.perf_counter() - start
        assert batch is not None
        sample_results.append(
            {
                "p_aug": p_aug,
                "batches": args.sample_batches,
                "seconds": seconds,
                "batches_per_sec": args.sample_batches / seconds,
                "batch_bytes": tree_nbytes(batch),
                "rss_mb_after": current_rss_mb(),
            }
        )
    results["sample_benchmark"] = sample_results

    pretraining_train.p_aug = 0.0
    batch_cpu = pretraining_train.sample(args.batch_size)
    with timed("device_put_one_batch", {"batch_bytes": tree_nbytes(batch_cpu)}):
        batch_device = jax.device_put(batch_cpu)
        block_tree(batch_device)

    example_batch = pretraining_train.sample(1)
    with timed("create_agent"):
        agent = CrossModalStateDistilledInFOMAgent.create(
            args.seed,
            example_batch["observations"],
            example_batch["actions"],
            config,
            ex_third_person_observations=example_batch["third_person_observations"],
            **state_stats,
        )
        block_tree(agent.network.params)

    with timed("compile_and_first_pretrain_update_device_batch"):
        agent, info = agent.pretrain(batch_device)
        block_tree(info)
    results["first_update_info"] = scalarize(info)

    start = time.perf_counter()
    for _ in range(args.update_steps):
        agent, info = agent.pretrain(batch_device)
        block_tree(info)
    seconds = time.perf_counter() - start
    results["device_batch_update_benchmark"] = {
        "updates": args.update_steps,
        "seconds": seconds,
        "updates_per_sec": args.update_steps / seconds,
        "last_info": scalarize(info),
        "rss_mb_after": current_rss_mb(),
    }

    start = time.perf_counter()
    for _ in range(args.update_steps):
        batch = pretraining_train.sample(args.batch_size)
        agent, info = agent.pretrain(batch)
        block_tree(info)
    seconds = time.perf_counter() - start
    results["sample_plus_update_benchmark"] = {
        "p_aug": pretraining_train.p_aug,
        "updates": args.update_steps,
        "seconds": seconds,
        "updates_per_sec": args.update_steps / seconds,
        "last_info": scalarize(info),
        "rss_mb_after": current_rss_mb(),
    }

    results["rss_mb_end"] = current_rss_mb()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
