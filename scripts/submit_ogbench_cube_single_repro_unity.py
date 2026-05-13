#!/usr/bin/env python3
"""Submit the OGBench cube-single InFOM paper reproduction campaign on Unity."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
from pathlib import Path


DEFAULT_BASE = Path('/work/pi_mengfanxu_umass_edu/bhuo_umass_edu')
DEFAULT_REPO = DEFAULT_BASE / 'repos/infom-learning-from-obs'
DEFAULT_PYTHON = DEFAULT_BASE / 'tools/miniforge3/envs/infom-obs/bin/python'
UPSTREAM_COMMIT = 'ed0761d7a349fb34b201071f98ac88b6d91cafe2'
JAX_PREFLIGHT_EXIT_CODE = 76
CHECKPOINT_SIGNAL_EXIT_CODE = 75


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


def parse_probe_specs(text: str) -> list[tuple[str, str, str]]:
    specs = []
    for raw in text.split(','):
        raw = raw.strip()
        if not raw:
            continue
        partition, constraint = raw.split(':', 1)
        safe_constraint = re.sub(r'[^A-Za-z0-9]+', '_', constraint).strip('_')
        name = f'{partition}_{safe_constraint}'
        specs.append((name, partition, constraint))
    return specs


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


def common_exports(repo: Path, python: Path, run_root: Path) -> str:
    return f"""RUN_ROOT={run_root}
REPO={repo}
PY={python}
DATA="$HOME/.ogbench/data"
export RUN_ROOT REPO PY DATA
export PYTHONPATH="$REPO"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_TRACEBACK_FILTERING=off
mkdir -p "$RUN_ROOT" "$DATA"
cd "$REPO"
"""


def expected_commit_check(expected_commit: str | None) -> str:
    if not expected_commit:
        return ''
    return f"""if [ "$(git rev-parse HEAD)" != "{expected_commit}" ]; then
  echo "Unexpected repo commit: $(git rev-parse HEAD), expected {expected_commit}" >&2
  exit 2
fi
"""


def jax_preflight(run_root: Path, label: str) -> str:
    return f""""$PY" scripts/jax_cuda_probe.py \\
  --output="{run_root}/probe_results/{label}_${{SLURM_JOB_ID}}.json"
code=$?
if [ "$code" != "0" ]; then
  echo "JAX CUDA preflight failed with code $code; requesting requeue." >&2
  exit {JAX_PREFLIGHT_EXIT_CODE}
fi
"""


def requeue_wrapper(command: str) -> str:
    return f"""set +e
(
set -e
{command}
)
code=$?
set -e
if [ "$code" = "{CHECKPOINT_SIGNAL_EXIT_CODE}" ] || [ "$code" = "{JAX_PREFLIGHT_EXIT_CODE}" ]; then
  echo "Requeueing ${{SLURM_JOB_ID}} after checkpoint/preflight exit code $code"
  scontrol requeue "${{SLURM_JOB_ID}}"
  exit 0
fi
exit "$code"
"""


def prepare_script(repo: Path, python: Path, run_root: Path, expected_commit: str | None = None) -> str:
    return f"""#!/bin/bash
#SBATCH --job-name=infom-repro-data
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output={run_root}/slurm/%x-%j.out
#SBATCH --error={run_root}/slurm/%x-%j.err

set -euo pipefail
{common_exports(repo, python, run_root)}
echo "host=$(hostname) job=${{SLURM_JOB_ID}} date=$(date -Is) repo=$(git rev-parse HEAD)"
{expected_commit_check(expected_commit)}
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

dataset_dir = os.path.expanduser("~/.ogbench/data")
os.makedirs(dataset_dir, exist_ok=True)
download_datasets(["cube-single-play-v0"], dataset_dir)
PY
if [ ! -f "$DATA/cube-single-play-ft-v0.npz" ] || [ ! -f "$DATA/cube-single-play-ft-v0-val.npz" ]; then
  "$PY" data_gen_scripts/generate_ogbench_manispace.py \\
    --env_name=cube-single-v0 \\
    --save_path="$DATA/cube-single-play-ft-v0.npz" \\
    --num_episodes=500 \\
    --max_episode_steps=1001 \\
    --dataset_type=play
fi
"$PY" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

import numpy as np

run_root = Path(os.environ["RUN_ROOT"])
dataset_dir = Path(os.path.expanduser("~/.ogbench/data"))
files = [
    "cube-single-play-v0.npz",
    "cube-single-play-v0-val.npz",
    "cube-single-play-ft-v0.npz",
    "cube-single-play-ft-v0-val.npz",
]
manifest = {{"dataset_dir": str(dataset_dir), "files": {{}}}}
for name in files:
    path = dataset_dir / name
    if not path.exists():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    info = {{"bytes": path.stat().st_size, "sha256": digest.hexdigest(), "arrays": {{}}}}
    with np.load(path) as data:
        for key in data.files:
            arr = data[key]
            info["arrays"][key] = {{"shape": list(arr.shape), "dtype": str(arr.dtype)}}
    manifest["files"][name] = info
out = run_root / "dataset_manifest.json"
out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\\n")
print(out)
PY
"""


def exclude_directive(exclude: str | None) -> str:
    return f'#SBATCH --exclude={exclude}\n' if exclude else ''


def probe_script(
    repo: Path,
    python: Path,
    run_root: Path,
    name: str,
    partition: str,
    constraint: str,
    exclude: str | None = None,
) -> str:
    return f"""#!/bin/bash
#SBATCH --job-name=infom-probe-{name}
#SBATCH --partition={partition}
#SBATCH --constraint={constraint}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
{exclude_directive(exclude).rstrip()}
#SBATCH --output={run_root}/slurm/%x-%j.out
#SBATCH --error={run_root}/slurm/%x-%j.err

set -euo pipefail
{common_exports(repo, python, run_root)}
echo "host=$(hostname) job=${{SLURM_JOB_ID}} partition={partition} constraint={constraint} date=$(date -Is) repo=$(git rev-parse HEAD)"
nvidia-smi
"$PY" scripts/jax_cuda_probe.py \\
  --output="{run_root}/probe_results/{name}_${{SLURM_JOB_ID}}.json"
"""


def smoke_script(
    repo: Path,
    python: Path,
    run_root: Path,
    exclude: str | None = None,
    partition: str = 'gpu',
    constraint: str = 'h100|a100',
    expected_commit: str | None = None,
) -> str:
    command = f"""{jax_preflight(run_root, 'smoke')}
"$PY" main.py \\
  --env_name=cube-single-play-singletask-task1-v0 \\
  --save_dir="$RUN_ROOT/smoke_runs" \\
  --wandb_run_group=unity_cube_single_paper_repro_smoke \\
  --run_id=smoke \\
  --enable_wandb=0 \\
  --enable_tensorboard=1 \\
  --resume_from_checkpoint=1 \\
  --checkpoint_interval=2 \\
  --checkpoint_keep=3 \\
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
  --agent.alpha=30 &
child=$!
forward_checkpoint_signal() {{
  echo "Forwarding checkpoint signal to Python child $child"
  kill -USR1 "$child" 2>/dev/null || true
}}
trap forward_checkpoint_signal USR1 TERM
wait "$child"
"""
    return f"""#!/bin/bash
#SBATCH --job-name=infom-repro-smoke
#SBATCH --partition={partition}
#SBATCH --constraint={constraint}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
{exclude_directive(exclude).rstrip()}
#SBATCH --output={run_root}/slurm/%x-%j.out
#SBATCH --error={run_root}/slurm/%x-%j.err

set -euo pipefail
{common_exports(repo, python, run_root)}
echo "host=$(hostname) job=${{SLURM_JOB_ID}} date=$(date -Is) repo=$(git rev-parse HEAD)"
{expected_commit_check(expected_commit)}
nvidia-smi
{requeue_wrapper(command)}
"""


def train_script(
    repo: Path,
    python: Path,
    run_root: Path,
    task: int,
    seed: int,
    exclude: str | None = None,
    partition: str = 'gpu',
    constraint: str = 'h100|a100',
    time_limit: str = '12:00:00',
    cpus: int = 8,
    mem: str = '32G',
    checkpoint_interval: int = 50_000,
    checkpoint_keep: int = 3,
    enable_tensorboard: int = 1,
    expected_commit: str | None = None,
) -> str:
    command = f"""{jax_preflight(run_root, f'train_t{task}_s{seed}')}
"$PY" main.py \\
  --env_name=cube-single-play-singletask-task${{TASK}}-v0 \\
  --save_dir="$RUN_ROOT/runs" \\
  --wandb_run_group="$GROUP" \\
  --run_id="task${{TASK}}_seed${{SEED}}" \\
  --enable_wandb=0 \\
  --enable_tensorboard={enable_tensorboard} \\
  --resume_from_checkpoint=1 \\
  --checkpoint_interval={checkpoint_interval} \\
  --checkpoint_keep={checkpoint_keep} \\
  --checkpoint_signal_exit_code={CHECKPOINT_SIGNAL_EXIT_CODE} \\
  --seed="$SEED" \\
  --agent=agents/infom.py \\
  --agent.expectile=0.95 \\
  --agent.kl_weight=0.05 \\
  --agent.alpha=30 &
child=$!
forward_checkpoint_signal() {{
  echo "Forwarding checkpoint signal to Python child $child"
  kill -USR1 "$child" 2>/dev/null || true
}}
trap forward_checkpoint_signal USR1 TERM
wait "$child"
"""
    return f"""#!/bin/bash
#SBATCH --job-name=infom-repro-t{task}-s{seed}
#SBATCH --partition={partition}
#SBATCH --constraint={constraint}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time_limit}
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
{exclude_directive(exclude).rstrip()}
#SBATCH --output={run_root}/slurm/%x-%j.out
#SBATCH --error={run_root}/slurm/%x-%j.err

set -euo pipefail
{common_exports(repo, python, run_root)}
TASK={task}
SEED={seed}
GROUP=unity_cube_single_paper_repro_task${{TASK}}
echo "host=$(hostname) job=${{SLURM_JOB_ID}} task=${{TASK}} seed=${{SEED}} date=$(date -Is) repo=$(git rev-parse HEAD)"
{expected_commit_check(expected_commit)}
nvidia-smi
{requeue_wrapper(command)}
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
    parser.add_argument('--run-root', type=Path, default=DEFAULT_BASE / f'infom-learning-from-obs/runs/unity_cube_single_paper_repro_{timestamp}')
    parser.add_argument('--repo', type=Path, default=DEFAULT_REPO)
    parser.add_argument('--python', type=Path, default=DEFAULT_PYTHON)
    parser.add_argument('--tasks', default='1-5')
    parser.add_argument('--seeds', default='0-3')
    parser.add_argument('--stage', choices=['all', 'prepare', 'probe', 'smoke', 'campaign'], default='all')
    parser.add_argument('--probe-specs', default='gpu:h100|a100,gpu:l40s,gpu-preempt:h100|a100,gpu-preempt:l40s')
    parser.add_argument('--partition', default='gpu,gpu-preempt')
    parser.add_argument('--constraint', default='h100|a100')
    parser.add_argument('--time-limit', default='12:00:00')
    parser.add_argument('--cpus-per-task', type=int, default=8)
    parser.add_argument('--mem', default='32G')
    parser.add_argument('--checkpoint-interval', type=int, default=50_000)
    parser.add_argument('--checkpoint-keep', type=int, default=3)
    parser.add_argument('--enable-tensorboard', type=int, default=1)
    parser.add_argument('--expected-commit', default=None)
    parser.add_argument('--exclude', help='Optional Slurm node list to exclude for GPU jobs, e.g. gpu026.')
    parser.add_argument('--no-submit', action='store_true')
    args = parser.parse_args()

    tasks = parse_range_list(args.tasks)
    seeds = parse_range_list(args.seeds)
    probe_specs = parse_probe_specs(args.probe_specs)
    run_root = args.run_root
    slurm_dir = run_root / 'slurm'
    slurm_dir.mkdir(parents=True, exist_ok=True)

    scripts: dict[str, Path] = {}
    if args.stage in {'all', 'prepare'}:
        scripts['prepare'] = slurm_dir / 'prepare_data.sbatch'
        write(scripts['prepare'], prepare_script(args.repo, args.python, run_root, args.expected_commit))
    if args.stage in {'all', 'probe'}:
        for name, partition, constraint in probe_specs:
            stage = f'probe_{name}'
            scripts[stage] = slurm_dir / f'{stage}.sbatch'
            write(scripts[stage], probe_script(args.repo, args.python, run_root, name, partition, constraint, args.exclude))
    if args.stage in {'all', 'smoke'}:
        scripts['smoke'] = slurm_dir / 'smoke.sbatch'
        write(
            scripts['smoke'],
            smoke_script(
                args.repo,
                args.python,
                run_root,
                args.exclude,
                args.partition,
                args.constraint,
                args.expected_commit,
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
                        args.repo,
                        args.python,
                        run_root,
                        task,
                        seed,
                        args.exclude,
                        args.partition,
                        args.constraint,
                        args.time_limit,
                        args.cpus_per_task,
                        args.mem,
                        args.checkpoint_interval,
                        args.checkpoint_keep,
                        args.enable_tensorboard,
                        args.expected_commit,
                    ),
                )

    manifest = {
        'created_at': dt.datetime.now(dt.timezone.utc).isoformat(),
        'run_root': str(run_root),
        'repo': str(args.repo),
        'python': str(args.python),
        'upstream_commit': UPSTREAM_COMMIT,
        'tasks': tasks,
        'seeds': seeds,
        'probe_specs': probe_specs,
        'stage': args.stage,
        'exclude': args.exclude,
        'partition': args.partition,
        'constraint': args.constraint,
        'time_limit': args.time_limit,
        'cpus_per_task': args.cpus_per_task,
        'mem': args.mem,
        'checkpoint_interval': args.checkpoint_interval,
        'checkpoint_keep': args.checkpoint_keep,
        'expected_commit': args.expected_commit,
        'scripts': {stage: str(path) for stage, path in scripts.items()},
    }
    (run_root / 'submission_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')

    if args.no_submit:
        print(f'Wrote scripts under {slurm_dir}')
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
        for stage in sorted(scripts):
            if not stage.startswith('probe_') or stage in existing:
                continue
            job_id = sbatch(scripts[stage])
            f.write(f'{stage}\t{job_id}\t{scripts[stage]}\n')
            f.flush()
            print(f'{stage} {job_id}')
        if 'smoke' in scripts and smoke_id is None:
            dep = f'afterok:{prepare_id}' if prepare_id else None
            smoke_id = sbatch(scripts['smoke'], dep)
            f.write(f'smoke\t{smoke_id}\t{scripts["smoke"]}\n')
            f.flush()
            print(f'smoke {smoke_id}')
        train_dependency = f'afterok:{smoke_id}' if smoke_id else None
        for stage in sorted(scripts):
            if not re.match(r'train_t\d+_s\d+$', stage) or stage in existing:
                continue
            job_id = sbatch(scripts[stage], train_dependency)
            f.write(f'{stage}\t{job_id}\t{scripts[stage]}\n')
            f.flush()
            print(f'{stage} {job_id}')

    print(f'run_root {run_root}')
    print(f'jobs {job_path}')


if __name__ == '__main__':
    main()
