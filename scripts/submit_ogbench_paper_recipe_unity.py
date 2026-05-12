#!/usr/bin/env python3
"""Submit paper-recipe OGBench cube-single InFOM replication jobs on Unity.

This runner intentionally uses clean upstream InFOM code for data generation and
training. It isolates ``~/.ogbench/data`` under the run root so that pretraining
data comes from the official OGBench downloader and fine-tuning data comes from
the upstream InFOM generator, without reusing locally generated datasets.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
from pathlib import Path


DEFAULT_BASE = Path('/work/pi_mengfanxu_umass_edu/bhuo_umass_edu')
DEFAULT_UPSTREAM_REPO = DEFAULT_BASE / 'repos/infom-upstream-ed0761d'
DEFAULT_PROBE_REPO = DEFAULT_BASE / 'repos/infom-learning-from-obs'
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


def common_exports(repo: Path, probe_repo: Path, python: Path, run_root: Path) -> str:
    return f"""RUN_ROOT={run_root}
REPO={repo}
PROBE_REPO={probe_repo}
PY={python}
export RUN_ROOT REPO PROBE_REPO PY
export HOME="$RUN_ROOT/home"
export DATA="$HOME/.ogbench/data"
export PYTHONPATH="$REPO"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_TRACEBACK_FILTERING=off
mkdir -p "$RUN_ROOT" "$HOME" "$DATA" "$RUN_ROOT/probe_results"
cd "$REPO"
"""


def expected_commit_check(expected_commit: str = UPSTREAM_COMMIT) -> str:
    return f"""if [ "$(git rev-parse HEAD)" != "{expected_commit}" ]; then
  echo "Unexpected repo commit: $(git rev-parse HEAD), expected {expected_commit}" >&2
  exit 2
fi
if [ -n "$(git status --porcelain)" ]; then
  echo "Upstream replication repo is dirty:" >&2
  git status --short >&2
  exit 2
fi
"""


def jax_preflight(run_root: Path, label: str) -> str:
    return f"""cd "$PROBE_REPO"
PYTHONPATH="$PROBE_REPO" "$PY" scripts/jax_cuda_probe.py \\
  --output="{run_root}/probe_results/{label}_${{SLURM_JOB_ID}}.json"
code=$?
if [ "$code" != "0" ]; then
  echo "JAX CUDA preflight failed with code $code." >&2
  exit {JAX_PREFLIGHT_EXIT_CODE}
fi
cd "$REPO"
export PYTHONPATH="$REPO"
"""


def env_report_python(label: str) -> str:
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

dataset_dir = Path(os.environ["DATA"])
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
    info = {{"bytes": path.stat().st_size, "sha256": digest.hexdigest(), "arrays": {{}}}}
    try:
        import numpy as np
        with np.load(path) as data:
            for key in data.files:
                arr = data[key]
                info["arrays"][key] = {{"shape": list(arr.shape), "dtype": str(arr.dtype)}}
    except Exception as exc:
        info["array_error"] = repr(exc)
    files[name] = info

out = {{
    "python": sys.version,
    "repo": os.environ["REPO"],
    "repo_head": os.popen("git rev-parse HEAD").read().strip(),
    "data_policy": "official OGBench play pretraining; upstream InFOM play-ft generation",
    "packages": packages,
    "dataset_dir": str(dataset_dir),
    "files": files,
}}
path = Path(os.environ["RUN_ROOT"]) / f"env_data_{label}_{{os.environ.get('SLURM_JOB_ID', 'local')}}.json"
path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\\n")
print(path)
PY
"""


def prepare_script(repo: Path, probe_repo: Path, python: Path, run_root: Path) -> str:
    return f"""#!/bin/bash
#SBATCH --job-name=infom-paper-data
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output={run_root}/slurm/%x-%j.out
#SBATCH --error={run_root}/slurm/%x-%j.err

set -euo pipefail
{common_exports(repo, probe_repo, python, run_root)}
echo "host=$(hostname) job=${{SLURM_JOB_ID}} stage=prepare date=$(date -Is) repo=$(git rev-parse HEAD)"
{expected_commit_check()}
"$PY" - <<'PY'
import importlib
import sys

mods = ["jax", "jaxlib", "flax", "ogbench", "numpy", "gymnasium", "ml_collections", "distrax"]
print(sys.version)
for name in mods:
    mod = importlib.import_module(name)
    print(f"{{name}}={{getattr(mod, '__version__', 'unknown')}}")
PY
"$PY" - <<'PY'
import os
from ogbench.utils import download_datasets

dataset_dir = os.environ["DATA"]
os.makedirs(dataset_dir, exist_ok=True)
download_datasets(["cube-single-play-v0"], dataset_dir)
PY
if [ ! -f "$DATA/cube-single-play-ft-v0.npz" ] || [ ! -f "$DATA/cube-single-play-ft-v0-val.npz" ]; then
  rm -f "$DATA/cube-single-play-ft-v0.npz" "$DATA/cube-single-play-ft-v0-val.npz"
  "$PY" data_gen_scripts/generate_ogbench_manispace.py \\
    --env_name=cube-single-v0 \\
    --save_path="$DATA/cube-single-play-ft-v0.npz" \\
    --num_episodes=500 \\
    --max_episode_steps=1001 \\
    --dataset_type=play
fi
{env_report_python('prepare')}
"""


def smoke_script(
    repo: Path,
    probe_repo: Path,
    python: Path,
    run_root: Path,
    partition: str,
    constraint: str,
    exclude: str | None,
) -> str:
    return f"""#!/bin/bash
#SBATCH --job-name=infom-paper-smoke
#SBATCH --partition={partition}
#SBATCH --constraint={constraint}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
{exclude_directive(exclude).rstrip()}
#SBATCH --output={run_root}/slurm/%x-%j.out
#SBATCH --error={run_root}/slurm/%x-%j.err

set -euo pipefail
{common_exports(repo, probe_repo, python, run_root)}
echo "host=$(hostname) job=${{SLURM_JOB_ID}} stage=smoke date=$(date -Is) repo=$(git rev-parse HEAD)"
{expected_commit_check()}
nvidia-smi
{jax_preflight(run_root, 'smoke')}
{env_report_python('smoke')}
"$PY" main.py \\
  --env_name=cube-single-play-singletask-task1-v0 \\
  --save_dir="$RUN_ROOT/smoke_runs" \\
  --wandb_run_group=paper_recipe_smoke \\
  --wandb_mode=offline \\
  --enable_wandb=0 \\
  --pretraining_steps=2 \\
  --finetuning_steps=2 \\
  --log_interval=1 \\
  --eval_interval=1 \\
  --eval_episodes=1 \\
  --save_interval=999999 \\
  --seed=0 \\
  --agent=agents/infom.py \\
  --agent.expectile=0.95 \\
  --agent.kl_weight=0.05 \\
  --agent.alpha=30
"""


def train_script(
    repo: Path,
    probe_repo: Path,
    python: Path,
    run_root: Path,
    task: int,
    seed: int,
    partition: str,
    constraint: str,
    exclude: str | None,
    cpus: int,
    mem: str,
    time_limit: str,
) -> str:
    label = f'train_t{task}_s{seed}'
    return f"""#!/bin/bash
#SBATCH --job-name=infom-paper-t{task}-s{seed}
#SBATCH --partition={partition}
#SBATCH --constraint={constraint}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time_limit}
{exclude_directive(exclude).rstrip()}
#SBATCH --output={run_root}/slurm/%x-%j.out
#SBATCH --error={run_root}/slurm/%x-%j.err

set -euo pipefail
{common_exports(repo, probe_repo, python, run_root)}
TASK={task}
SEED={seed}
GROUP=paper_recipe_cube_single_task${{TASK}}
echo "host=$(hostname) job=${{SLURM_JOB_ID}} stage=train task=${{TASK}} seed=${{SEED}} date=$(date -Is) repo=$(git rev-parse HEAD)"
{expected_commit_check()}
nvidia-smi
{jax_preflight(run_root, label)}
{env_report_python(label)}
"$PY" main.py \\
  --env_name=cube-single-play-singletask-task${{TASK}}-v0 \\
  --save_dir="$RUN_ROOT/runs/task${{TASK}}_seed${{SEED}}" \\
  --wandb_run_group="$GROUP" \\
  --wandb_mode=offline \\
  --enable_wandb=0 \\
  --seed="$SEED" \\
  --agent=agents/infom.py \\
  --agent.expectile=0.95 \\
  --agent.kl_weight=0.05 \\
  --agent.alpha=30
"""


def parse_job_ids(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    jobs: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith('stage\t'):
            continue
        stage, job_id, *_ = line.split('\t')
        jobs[stage] = job_id
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    parser.add_argument(
        '--run-root',
        type=Path,
        default=DEFAULT_BASE / f'infom-learning-from-obs/runs/unity_cube_single_paper_recipe_{timestamp}',
    )
    parser.add_argument('--upstream-repo', type=Path, default=DEFAULT_UPSTREAM_REPO)
    parser.add_argument('--probe-repo', type=Path, default=DEFAULT_PROBE_REPO)
    parser.add_argument('--python', type=Path, default=DEFAULT_PYTHON)
    parser.add_argument('--tasks', default='1,4,5')
    parser.add_argument('--seeds', default='0-3')
    parser.add_argument('--stage', choices=['all', 'prepare', 'smoke', 'campaign'], default='all')
    parser.add_argument('--partition', default='gpu')
    parser.add_argument('--constraint', default='a100')
    parser.add_argument('--time-limit', default='12:00:00')
    parser.add_argument('--cpus-per-task', type=int, default=8)
    parser.add_argument('--mem', default='32G')
    parser.add_argument('--exclude')
    parser.add_argument('--no-submit', action='store_true')
    args = parser.parse_args()

    tasks = parse_range_list(args.tasks)
    seeds = parse_range_list(args.seeds)
    run_root = args.run_root
    slurm_dir = run_root / 'slurm'
    slurm_dir.mkdir(parents=True, exist_ok=True)

    scripts: dict[str, Path] = {}
    if args.stage in {'all', 'prepare'}:
        scripts['prepare'] = slurm_dir / 'prepare_data.sbatch'
        write(scripts['prepare'], prepare_script(args.upstream_repo, args.probe_repo, args.python, run_root))
    if args.stage in {'all', 'smoke'}:
        scripts['smoke'] = slurm_dir / 'smoke.sbatch'
        write(
            scripts['smoke'],
            smoke_script(
                args.upstream_repo,
                args.probe_repo,
                args.python,
                run_root,
                args.partition,
                args.constraint,
                args.exclude,
            ),
        )
    if args.stage in {'all', 'campaign'}:
        for task in tasks:
            for seed in seeds:
                stage = f'train_t{task}_s{seed}'
                scripts[stage] = slurm_dir / f'{stage}.sbatch'
                write(
                    scripts[stage],
                    train_script(
                        args.upstream_repo,
                        args.probe_repo,
                        args.python,
                        run_root,
                        task,
                        seed,
                        args.partition,
                        args.constraint,
                        args.exclude,
                        args.cpus_per_task,
                        args.mem,
                        args.time_limit,
                    ),
                )

    manifest = {
        'created_at': dt.datetime.now(dt.timezone.utc).isoformat(),
        'run_root': str(run_root),
        'upstream_repo': str(args.upstream_repo),
        'upstream_commit': UPSTREAM_COMMIT,
        'probe_repo': str(args.probe_repo),
        'python': str(args.python),
        'data_home': str(run_root / 'home'),
        'data_policy': 'official OGBench cube-single-play-v0 pretraining; upstream InFOM cube-single-play-ft-v0 generation',
        'tasks': tasks,
        'seeds': seeds,
        'stage': args.stage,
        'partition': args.partition,
        'constraint': args.constraint,
        'time_limit': args.time_limit,
        'cpus_per_task': args.cpus_per_task,
        'mem': args.mem,
        'exclude': args.exclude,
        'scripts': {stage: str(path) for stage, path in scripts.items()},
    }
    (run_root / 'submission_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')

    if args.no_submit:
        print(f'Wrote scripts under {slurm_dir}')
        print(run_root)
        return

    job_path = run_root / 'jobs.tsv'
    existing = parse_job_ids(job_path)
    with job_path.open('a') as f:
        if job_path.stat().st_size == 0:
            f.write('stage\tjob_id\tscript\n')

        prepare_id = existing.get('prepare')
        smoke_id = existing.get('smoke')
        if 'prepare' in scripts and prepare_id is None:
            prepare_id = sbatch(scripts['prepare'])
            f.write(f'prepare\t{prepare_id}\t{scripts["prepare"]}\n')
            f.flush()
            print(f'prepare {prepare_id}')
        if 'smoke' in scripts and smoke_id is None:
            dep = f'afterok:{prepare_id}' if prepare_id else None
            smoke_id = sbatch(scripts['smoke'], dep)
            f.write(f'smoke\t{smoke_id}\t{scripts["smoke"]}\n')
            f.flush()
            print(f'smoke {smoke_id}')
        train_dependency = f'afterok:{smoke_id}' if smoke_id else (f'afterok:{prepare_id}' if prepare_id else None)
        for stage in sorted(scripts):
            if not re.fullmatch(r'train_t\d+_s\d+', stage) or stage in existing:
                continue
            job_id = sbatch(scripts[stage], train_dependency)
            f.write(f'{stage}\t{job_id}\t{scripts[stage]}\n')
            f.flush()
            print(f'{stage} {job_id}')

    print(f'run_root {run_root}')
    print(f'jobs {job_path}')


if __name__ == '__main__':
    main()
