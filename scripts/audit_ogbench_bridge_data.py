#!/usr/bin/env python3
"""Audit paired OGBench bridge datasets.

The bridge files should contain one synchronized ego/state observation and one
third-person RGB render per transition. Task rewards are checked through the
normal OGBench relabeling path, so a single cube-single play dataset can back
task1-task5 fine-tuning views.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs import ogbench_bridge_utils


REQUIRED_KEYS = (
    "observations",
    "third_person_observations",
    "actions",
    "terminals",
    "qpos",
    "qvel",
    "episode_ids",
    "timesteps",
)


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonify(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


def _fail(report: dict[str, Any], message: str) -> None:
    report.setdefault("errors", []).append(message)


def _warn(report: dict[str, Any], message: str) -> None:
    report.setdefault("warnings", []).append(message)


def _sample_indices(size: int, count: int) -> np.ndarray:
    if size <= 0:
        return np.array([], dtype=np.int64)
    count = min(count, size)
    return np.unique(np.linspace(0, size - 1, num=count, dtype=np.int64))


def audit_raw_file(path: Path, sample_checks: int) -> dict[str, Any]:
    report: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "errors": [],
        "warnings": [],
    }
    if not path.exists():
        _fail(report, "file does not exist")
        return report

    report["bytes"] = path.stat().st_size
    with np.load(path) as raw:
        keys = sorted(raw.files)
        report["keys"] = keys
        missing = [key for key in REQUIRED_KEYS if key not in raw]
        if missing:
            _fail(report, f"missing required keys: {missing}")
            return report

        lengths = {key: int(raw[key].shape[0]) for key in keys}
        report["lengths"] = lengths
        expected_len = lengths["observations"]
        bad_lengths = {key: val for key, val in lengths.items() if val != expected_len}
        if bad_lengths:
            _fail(report, f"leading dimensions do not match observations: {bad_lengths}")

        observations = raw["observations"]
        third_person = raw["third_person_observations"]
        actions = raw["actions"]
        terminals = raw["terminals"].astype(bool)
        episode_ids = raw["episode_ids"]
        timesteps = raw["timesteps"]

        report["observations_shape"] = tuple(int(v) for v in observations.shape)
        report["third_person_shape"] = tuple(int(v) for v in third_person.shape)
        report["actions_shape"] = tuple(int(v) for v in actions.shape)
        report["observations_dtype"] = str(observations.dtype)
        report["third_person_dtype"] = str(third_person.dtype)
        report["actions_dtype"] = str(actions.dtype)
        report["terminal_count"] = int(terminals.sum())
        report["episode_count"] = int(len(np.unique(episode_ids))) if len(episode_ids) else 0
        report["first_episode_id"] = int(episode_ids[0]) if len(episode_ids) else None
        report["last_episode_id"] = int(episode_ids[-1]) if len(episode_ids) else None

        if third_person.dtype != np.uint8:
            _fail(report, f"third_person_observations dtype is {third_person.dtype}, expected uint8")
        if third_person.ndim != 4 or third_person.shape[-1] != 3:
            _fail(report, "third_person_observations must have shape [N, H, W, 3]")
        if observations.ndim != 2:
            _warn(report, f"observations rank is {observations.ndim}, expected state-control rank 2")
        if actions.ndim != 2:
            _warn(report, f"actions rank is {actions.ndim}, expected rank 2")

        if len(episode_ids):
            boundaries = np.flatnonzero(episode_ids[1:] != episode_ids[:-1]) + 1
            starts = np.concatenate([[0], boundaries])
            ends = np.concatenate([boundaries, [len(episode_ids)]])
            if not np.all(timesteps[starts] == 0):
                _fail(report, "not every episode starts at timestep 0")
            prev_ends = boundaries - 1
            if len(prev_ends) and not np.all(terminals[prev_ends]):
                _fail(report, "episode id changes before a terminal transition")
            if not np.all(terminals[ends - 1]):
                _fail(report, "not every episode ends with a terminal transition")

            sample_eps = _sample_indices(len(starts), sample_checks)
            bad_timestep_eps = []
            for idx in sample_eps:
                start = starts[idx]
                end = ends[idx]
                expected = np.arange(end - start, dtype=timesteps.dtype)
                if not np.array_equal(timesteps[start:end], expected):
                    bad_timestep_eps.append(int(episode_ids[start]))
            if bad_timestep_eps:
                _fail(report, f"sampled episodes have non-contiguous timesteps: {bad_timestep_eps[:10]}")

    return report


def audit_loaded_alignment(path: Path, sample_checks: int) -> dict[str, Any]:
    report: dict[str, Any] = {
        "path": str(path),
        "errors": [],
        "warnings": [],
    }
    if not path.exists():
        _fail(report, "file does not exist")
        return report

    loaded = ogbench_bridge_utils.load_dataset(str(path))
    with np.load(path) as raw:
        terminals = raw["terminals"].astype(bool)
        valid_idxs = np.flatnonzero(~terminals)
        report["loaded_size"] = int(loaded["observations"].shape[0])
        report["expected_loaded_size"] = int(valid_idxs.shape[0])
        if loaded["observations"].shape[0] != valid_idxs.shape[0]:
            _fail(report, "loaded transition count does not match non-terminal raw transitions")
            return report

        for key in ("observations", "third_person_observations", "actions", "qpos", "qvel", "episode_ids", "timesteps"):
            if loaded[key].shape[0] != len(valid_idxs):
                _fail(report, f"loaded key {key} has wrong length {loaded[key].shape[0]}")
        for key in ("next_observations", "next_third_person_observations", "terminals"):
            if loaded[key].shape[0] != len(valid_idxs):
                _fail(report, f"loaded key {key} has wrong length {loaded[key].shape[0]}")

        sample = _sample_indices(len(valid_idxs), sample_checks)
        raw_current = valid_idxs[sample]
        raw_next = raw_current + 1
        checks = {
            "observations": (loaded["observations"][sample], raw["observations"][raw_current]),
            "next_observations": (loaded["next_observations"][sample], raw["observations"][raw_next]),
            "third_person_observations": (
                loaded["third_person_observations"][sample],
                raw["third_person_observations"][raw_current],
            ),
            "next_third_person_observations": (
                loaded["next_third_person_observations"][sample],
                raw["third_person_observations"][raw_next],
            ),
            "actions": (loaded["actions"][sample], raw["actions"][raw_current]),
            "terminals": (loaded["terminals"][sample].astype(bool), raw["terminals"][raw_next].astype(bool)),
        }
        for key, (actual, expected) in checks.items():
            if not np.array_equal(actual, expected):
                _fail(report, f"loaded {key} is not aligned with raw transition sequence")

    return report


def summarize_task_rewards(dataset_dir: Path, task: int, split: str) -> dict[str, Any]:
    if split == "pretrain":
        dataset_name = f"bridge-cube-single-play-singletask-task{task}-v0"
    elif split == "finetune":
        dataset_name = f"bridge-cube-single-play-ft-singletask-task{task}-v0"
    else:
        raise ValueError(f"unknown split: {split}")

    _, train_dataset, val_dataset = ogbench_bridge_utils.make_env_and_datasets(
        dataset_name,
        dataset_dir=str(dataset_dir),
    )

    def one(name: str, dataset: dict[str, np.ndarray]) -> dict[str, Any]:
        item: dict[str, Any] = {"size": int(dataset["observations"].shape[0])}
        rewards = dataset.get("rewards")
        if rewards is not None:
            item.update(
                reward_min=float(np.min(rewards)),
                reward_max=float(np.max(rewards)),
                reward_mean=float(np.mean(rewards)),
                reward_ge_zero_frac=float(np.mean(rewards >= 0.0)),
            )
        terminals = dataset.get("terminals")
        if terminals is not None:
            item["terminal_frac"] = float(np.mean(terminals > 0))
        masks = dataset.get("masks")
        if masks is not None:
            item["mask_mean"] = float(np.mean(masks))
        return {name: item}

    summary = {
        "task": task,
        "split": split,
        "dataset_name": dataset_name,
    }
    summary.update(one("train", train_dataset))
    summary.update(one("val", val_dataset))
    return summary


def parse_tasks(value: str) -> list[int]:
    tasks = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        tasks.append(int(part))
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("~/.ogbench/data"))
    parser.add_argument("--tasks", default="1,2,3,4,5")
    parser.add_argument(
        "--task-splits",
        choices=("none", "pretrain", "finetune", "both"),
        default="both",
        help="Which logical task-relabel views to load and summarize.",
    )
    parser.add_argument("--sample-checks", type=int, default=16)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--fail-on-warning", action="store_true")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.expanduser().resolve()
    physical_files = [
        "bridge-cube-single-play-v0.npz",
        "bridge-cube-single-play-v0-val.npz",
        "bridge-cube-single-play-ft-v0.npz",
        "bridge-cube-single-play-ft-v0-val.npz",
    ]

    report: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "raw_files": {},
        "loaded_alignment": {},
        "task_rewards": [],
        "errors": [],
        "warnings": [],
    }

    for name in physical_files:
        path = dataset_dir / name
        raw_report = audit_raw_file(path, args.sample_checks)
        report["raw_files"][name] = raw_report
        report["errors"].extend(f"{name}: {msg}" for msg in raw_report.get("errors", []))
        report["warnings"].extend(f"{name}: {msg}" for msg in raw_report.get("warnings", []))

        alignment_report = audit_loaded_alignment(path, args.sample_checks)
        report["loaded_alignment"][name] = alignment_report
        report["errors"].extend(f"{name}: {msg}" for msg in alignment_report.get("errors", []))
        report["warnings"].extend(f"{name}: {msg}" for msg in alignment_report.get("warnings", []))

    if args.task_splits != "none":
        split_names = ("pretrain", "finetune") if args.task_splits == "both" else (args.task_splits,)
        for split in split_names:
            for task in parse_tasks(args.tasks):
                try:
                    report["task_rewards"].append(summarize_task_rewards(dataset_dir, task, split))
                except Exception as exc:  # noqa: BLE001 - report all audit failures together.
                    report["errors"].append(f"{split} task{task}: {type(exc).__name__}: {exc}")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(_jsonify(report), indent=2, sort_keys=True) + "\n")

    print(json.dumps(_jsonify(report), indent=2, sort_keys=True))

    has_errors = bool(report["errors"])
    has_warnings = bool(report["warnings"])
    if has_errors or (args.fail_on_warning and has_warnings):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
