#!/usr/bin/env python3
"""Save a small visual sanity grid for state-distilled bridge batches."""

import argparse
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from envs.ogbench_bridge_utils import load_dataset
from utils.datasets import Dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', required=True, help='Path to a bridge-*.npz file.')
    parser.add_argument('--output_path', required=True, help='PNG path for the sample grid.')
    parser.add_argument('--num_samples', type=int, default=8)
    parser.add_argument('--rgb_frame_stack', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    raw_dataset = load_dataset(args.dataset_path)
    dataset = Dataset.create(**raw_dataset)
    dataset.obs_norm_type = 'none'
    dataset.return_next_actions = True
    dataset.aux_frame_stack = args.rgb_frame_stack
    dataset.aux_frame_stack_keys = ('third_person_observations',)
    dataset.normalize_observations()

    idxs = rng.integers(0, dataset.size, size=args.num_samples)
    batch = dataset.sample(args.num_samples, idxs=idxs)

    cols = 2
    rows = args.num_samples
    fig, axes = plt.subplots(rows, cols, figsize=(8, max(2.2 * rows, 3.0)))
    if rows == 1:
        axes = axes[None, :]

    for row in range(rows):
        rgb = batch['third_person_observations'][row][..., -3:]
        next_rgb = batch['next_third_person_observations'][row][..., -3:]
        state = batch['observations'][row]
        next_state = batch['next_observations'][row]
        action = batch['actions'][row]
        next_action = batch['next_actions'][row]
        reward_text = ''
        if 'rewards' in batch:
            reward_text = f"\nr={float(batch['rewards'][row]):.3f}"

        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(
            f"idx={int(idxs[row])} state[:5]={np.round(state[:5], 3)}\n"
            f"a={np.round(action, 3)}{reward_text}",
            fontsize=8,
        )
        axes[row, 0].axis('off')

        axes[row, 1].imshow(next_rgb)
        axes[row, 1].set_title(
            f"next state[:5]={np.round(next_state[:5], 3)}\n"
            f"next a={np.round(next_action, 3)} terminal={float(batch['terminals'][row]):.0f}",
            fontsize=8,
        )
        axes[row, 1].axis('off')

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output_path, dpi=160)
    print(f'Saved bridge batch inspection grid to {args.output_path}')


if __name__ == '__main__':
    main()
