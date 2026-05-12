#!/usr/bin/env python3
"""Aggregate chunked OGBench bridge datasets into loader-facing files."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as file:
        return {key: file[key][...] for key in file.files}


def offset_episode_ids(dataset: dict[str, np.ndarray], offset: int) -> tuple[dict[str, np.ndarray], int]:
    dataset = {key: value for key, value in dataset.items()}
    if "episode_ids" not in dataset:
        raise KeyError("Missing episode_ids in chunk.")
    episode_ids = dataset["episode_ids"].astype(np.int64, copy=True) + offset
    dataset["episode_ids"] = episode_ids.astype(np.int32)
    next_offset = int(episode_ids.max()) + 1 if len(episode_ids) else offset
    return dataset, next_offset


def concatenate(paths: list[Path]) -> dict[str, np.ndarray]:
    if not paths:
        raise ValueError("No chunk files matched.")

    pieces = []
    episode_offset = 0
    for path in paths:
        dataset, episode_offset = offset_episode_ids(load_npz(path), episode_offset)
        pieces.append(dataset)

    keys = pieces[0].keys()
    for path, piece in zip(paths, pieces):
        if piece.keys() != keys:
            raise KeyError(f"Chunk keys differ in {path}: {sorted(piece.keys())} vs {sorted(keys)}")

    return {key: np.concatenate([piece[key] for piece in pieces], axis=0) for key in keys}


def save(path: Path, dataset: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **dataset)
    print(f"Saved {path} with {len(dataset['observations'])} transitions")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True, help="Output prefix, e.g. bridge-cube-single-play-v0")
    parser.add_argument("--chunk-glob", required=True, help="Glob relative to chunk-dir for train chunk files.")
    parser.add_argument("--val-chunk-glob", required=True, help="Glob relative to chunk-dir for val chunk files.")
    args = parser.parse_args()

    chunk_dir = args.chunk_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    train_paths = sorted(Path(path) for path in glob.glob(str(chunk_dir / args.chunk_glob)))
    val_paths = sorted(Path(path) for path in glob.glob(str(chunk_dir / args.val_chunk_glob)))

    print("Train chunks:")
    for path in train_paths:
        print(f"  {path}")
    print("Val chunks:")
    for path in val_paths:
        print(f"  {path}")

    train_dataset = concatenate(train_paths)
    val_dataset = concatenate(val_paths)
    save(output_dir / f"{args.prefix}.npz", train_dataset)
    save(output_dir / f"{args.prefix}-val.npz", val_dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
