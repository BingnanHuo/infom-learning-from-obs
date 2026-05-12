#!/usr/bin/env python3
"""Submit controlled upstream-vs-current OGBench InFOM A/B jobs on Unity."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
from pathlib import Path


DEFAULT_BASE = Path('/work/pi_mengfanxu_umass_edu/bhuo_umass_edu')
DEFAULT_CURRENT_REPO = DEFAULT_BASE / 'repos/infom-learning-from-obs'
DEFAULT_UPSTREAM_REPO = DEFAULT_BASE / 'repos/infom-upstream-ed0761d'
DEFAULT_PYTHON = DEFAULT_BASE / 'tools/miniforge3/envs/infom-obs/bin/python'
UPSTREAM_COMMIT = 'ed0761d7a349fb34b201071f98ac88b6d91cafe2'
JAX_PREFLIGHT_EXIT_CODE = 76


def run(cmd: list[str], *, cwd: Path | None = None) -> str:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc.stdout.strip()


def sbatch(script: Path, dependency: str | None = None) -> str:
    cmd = ['sbatch', '--parsable']
    if dependency:
        cmd.append(f'--dependency={dependency}')
    cmd.append(str(script))
    output = run(cmd)
    for line in reversed(output.splitlines()):
        candidate = line.split(';', 1)[0].strip()
        if re.fullmatch(r'\d+(?:_\d+)?', candidate):
            return candidate
    raise RuntimeError(f'Could not parse sbatch job id from output:\n{output}')


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)


def parse_range_list(text: str) -> list[int]:
    values: list[int] = []
    for part in text.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            start, end = part.split('-', 1)
            values.extend(range(int(start), int(end) + 1))
        else:
            values.append(int(part))
    return values


def exclude_directive(exclude: str | None) -> str:
    return f'#SBATCH --exclude={exclude}\n' if exclude else ''


def common_exports(target_repo: Path, probe_repo: Path, python: Path, run_root: Path) -> str:
    return f"""RUN_ROOT={run_root}
REPO={target_repo}
PROBE_REPO={probe_repo}
PY={python}
DATA="$HOME/.ogbench/data"
export RUN_ROOT REPO PROBE_REPO PY DATA
export PYTHONPATH="$REPO"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_TRACEBACK_FILTERING=off
mkdir -p "$RUN_ROOT" "$DATA"
"""


def expected_commit_check(expected_commit: str | None) -> str:
    if not expected_commit:
        return ''
    return f"""if [ "$(git rev-parse HEAD)" != "{expected_commit}" ]; then
  echo "Unexpected repo commit: $(git rev-parse HEAD), expected {expected_commit}" >&2
  exit 2
fi
"""


def preflight(run_root: Path, label: str) -> str:
    return f"""cd "$PROBE_REPO"
"$PY" scripts/jax_cuda_probe.py \\
  --output="{run_root}/probe_results/{label}_${{SLURM_JOB_ID}}.json"
code=$?
if [ "$code" != "0" ]; then
  echo "JAX CUDA preflight failed with code $code; exiting for requeue/debug." >&2
  exit {JAX_PREFLIGHT_EXIT_CODE}
fi
cd "$REPO"
"""


def package_and_data_report(run_root: Path, label: str) -> str:
    return f""""$PY" - <<'PY'
import hashlib
import json
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

packages = {{}}
for name in ["jax", "jaxlib", "flax", "optax", "ogbench", "numpy", "gymnasium", "distrax", "ml_collections", "mujoco", "dm_control"]:
    try:
        packages[name] = version(name)
    except PackageNotFoundError:
        packages[name] = None

dataset_dir = Path(os.path.expanduser("~/.ogbench/data"))
files = {{}}
for name in [
    "cube-single-play-v0.npz",
    "cube-single-play-v0-val.npz",
    "cube-single-play-ft-v0.npz",
    "cube-single-play-ft-v0-val.npz",
]:
    path = dataset_dir / name
    if not path.exists():
        files[name] = {{"missing": True}}
        continue
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    files[name] = {{"bytes": path.stat().st_size, "sha256": digest.hexdigest()}}

out = {{
    "python": sys.version,
    "repo": os.environ["REPO"],
    "packages": packages,
    "dataset_dir": str(dataset_dir),
    "files": files,
}}
path = Path("{run_root}") / "env_data_{label}_${{SLURM_JOB_ID}}.json"
path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\\n")
print(path)
PY
"""


def main_command(
    variant: str,
    run_root: Path,
    task: int,
    seed: int,
    current_variant: bool,
) -> str:
    extra = ''
    if current_variant:
        extra = """  --enable_tensorboard=0 \\
  --resume_from_checkpoint=0 \\
  --checkpoint_interval=0 \\
  --checkpoint_at_end=0 \\
  --checkpoint_on_signal=0 \\
"""
    return f""""$PY" main.py \\
  --env_name=cube-single-play-singletask-task{task}-v0 \\
  --save_dir="{run_root}/runs/{variant}_task{task}_seed{seed}" \\
  --wandb_run_group=ab_{variant}_task{task} \\
  --wandb_mode=offline \\
  --enable_wandb=0 \\
  --seed={seed} \\
{extra}  --agent=agents/infom.py \\
  --agent.expectile=0.95 \\
  --agent.kl_weight=0.05 \\
  --agent.alpha=30
"""


def train_script(
    *,
    variant: str,
    target_repo: Path,
    expected_commit: str | None,
    probe_repo: Path,
    python: Path,
    run_root: Path,
    task: int,
    seed: int,
    partition: str,
    constraint: str,
    exclude: str | None,
    cpus_per_task: int,
    mem: str,
    time_limit: str,
    current_variant: bool,
) -> str:
    label = f'{variant}_t{task}_s{seed}'
    return f"""#!/bin/bash
#SBATCH --job-name=infom-ab-{variant}-t{task}-s{seed}
#SBATCH --partition={partition}
#SBATCH --constraint={constraint}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --time={time_limit}
{exclude_directive(exclude).rstrip()}
#SBATCH --output={run_root}/slurm/%x-%j.out
#SBATCH --error={run_root}/slurm/%x-%j.err

set -euo pipefail
{common_exports(target_repo, probe_repo, python, run_root)}
cd "$REPO"
echo "host=$(hostname) job=${{SLURM_JOB_ID}} variant={variant} task={task} seed={seed} date=$(date -Is) repo=$(git rev-parse HEAD)"
{expected_commit_check(expected_commit)}
git status --short --branch
git diff --stat -- main.py agents/infom.py envs/env_utils.py utils/flax_utils.py utils/log_utils.py || true
nvidia-smi
{preflight(run_root, label)}
{package_and_data_report(run_root, label)}
{main_command(variant, run_root, task, seed, current_variant)}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    parser.add_argument('--run-root', type=Path, default=DEFAULT_BASE / f'infom-learning-from-obs/runs/unity_cube_single_repro_ab_{timestamp}')
    parser.add_argument('--current-repo', type=Path, default=DEFAULT_CURRENT_REPO)
    parser.add_argument('--upstream-repo', type=Path, default=DEFAULT_UPSTREAM_REPO)
    parser.add_argument('--probe-repo', type=Path, default=DEFAULT_CURRENT_REPO)
    parser.add_argument('--python', type=Path, default=DEFAULT_PYTHON)
    parser.add_argument('--tasks', default='1')
    parser.add_argument('--seeds', default='0')
    parser.add_argument('--partition', default='gpu-preempt')
    parser.add_argument('--constraint', default='h100')
    parser.add_argument('--time-limit', default='12:00:00')
    parser.add_argument('--cpus-per-task', type=int, default=8)
    parser.add_argument('--mem', default='32G')
    parser.add_argument('--exclude', default='gpu026,gpu030')
    parser.add_argument('--current-expected-commit')
    parser.add_argument('--upstream-expected-commit', default=UPSTREAM_COMMIT)
    parser.add_argument('--variants', default='upstream,current_noextras')
    parser.add_argument('--no-submit', action='store_true')
    args = parser.parse_args()

    tasks = parse_range_list(args.tasks)
    seeds = parse_range_list(args.seeds)
    variants = [value.strip() for value in args.variants.split(',') if value.strip()]
    run_root = args.run_root
    slurm_dir = run_root / 'slurm'
    slurm_dir.mkdir(parents=True, exist_ok=True)

    scripts: dict[str, Path] = {}
    for task in tasks:
        for seed in seeds:
            if 'upstream' in variants:
                stage = f'upstream_t{task}_s{seed}'
                scripts[stage] = slurm_dir / f'{stage}.sbatch'
                write(
                    scripts[stage],
                    train_script(
                        variant='upstream',
                        target_repo=args.upstream_repo,
                        expected_commit=args.upstream_expected_commit,
                        probe_repo=args.probe_repo,
                        python=args.python,
                        run_root=run_root,
                        task=task,
                        seed=seed,
                        partition=args.partition,
                        constraint=args.constraint,
                        exclude=args.exclude,
                        cpus_per_task=args.cpus_per_task,
                        mem=args.mem,
                        time_limit=args.time_limit,
                        current_variant=False,
                    ),
                )
            if 'current_noextras' in variants:
                stage = f'current_noextras_t{task}_s{seed}'
                scripts[stage] = slurm_dir / f'{stage}.sbatch'
                write(
                    scripts[stage],
                    train_script(
                        variant='current_noextras',
                        target_repo=args.current_repo,
                        expected_commit=args.current_expected_commit,
                        probe_repo=args.probe_repo,
                        python=args.python,
                        run_root=run_root,
                        task=task,
                        seed=seed,
                        partition=args.partition,
                        constraint=args.constraint,
                        exclude=args.exclude,
                        cpus_per_task=args.cpus_per_task,
                        mem=args.mem,
                        time_limit=args.time_limit,
                        current_variant=True,
                    ),
                )

    manifest = {
        'created_at': dt.datetime.now(dt.timezone.utc).isoformat(),
        'run_root': str(run_root),
        'current_repo': str(args.current_repo),
        'upstream_repo': str(args.upstream_repo),
        'probe_repo': str(args.probe_repo),
        'python': str(args.python),
        'tasks': tasks,
        'seeds': seeds,
        'variants': variants,
        'partition': args.partition,
        'constraint': args.constraint,
        'time_limit': args.time_limit,
        'exclude': args.exclude,
        'scripts': {key: str(value) for key, value in scripts.items()},
        'jobs': {},
    }

    if not args.no_submit:
        for stage, script in scripts.items():
            manifest['jobs'][stage] = sbatch(script)

    manifest_path = run_root / 'submission_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
