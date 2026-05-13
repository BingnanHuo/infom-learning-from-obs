#!/usr/bin/env python3
"""Submit state-only InFOM runs on a newly collected OGBench bridge dataset."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path


DEFAULT_BASE = Path("/work/pi_mengfanxu_umass_edu/bhuo_umass_edu")
DEFAULT_REPO = DEFAULT_BASE / "repos/infom-learning-from-obs"
DEFAULT_PYTHON = DEFAULT_BASE / "tools/miniforge3/envs/infom-obs/bin/python"
DEFAULT_DATA_ROOT = (
    DEFAULT_BASE
    / "infom-learning-from-obs/data/bridge_cube_single_play_20260506_004058"
)
CHECKPOINT_SIGNAL_EXIT_CODE = 75
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


def remote_run(host: str, command: str) -> str:
    return run(["ssh", host, command])


def remote_copy(host: str, source: Path, dest: Path) -> None:
    run(["scp", str(source), f"{host}:{dest}"])


def sbatch(host: str, script: Path, dependency: str | None = None) -> str:
    cmd = ["sbatch", "--parsable"]
    if dependency:
        cmd.append(f"--dependency={dependency}")
    cmd.append(str(script))
    output = remote_run(host, " ".join(shlex.quote(part) for part in cmd))
    for line in reversed(output.splitlines()):
        candidate = line.split(";", 1)[0].strip()
        if re.fullmatch(r"\d+(?:_\d+)?", candidate):
            return candidate
    raise RuntimeError(f"Could not parse sbatch job id from output:\n{output}")


def parse_range_list(text: str) -> list[int]:
    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            values.extend(range(int(start), int(end) + 1))
        else:
            values.append(int(part))
    return values


def optional_sbatch(name: str, value: str | None) -> str:
    if value is None or value == "":
        return ""
    return f"#SBATCH --{name}={value}\n"


def common_exports(repo: Path, python: Path, run_root: Path, data_root: Path) -> str:
    return f"""RUN_ROOT={run_root}
REPO={repo}
PY={python}
DATA_ROOT={data_root}
export RUN_ROOT REPO PY DATA_ROOT
export PYTHONPATH="$REPO"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_TRACEBACK_FILTERING=off
mkdir -p "$RUN_ROOT" "$RUN_ROOT/probe_results"
cd "$REPO"
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


def smoke_script(
    repo: Path,
    python: Path,
    run_root: Path,
    data_root: Path,
    partition: str,
    constraint: str,
    cpus: int,
    mem: str,
    gres: str,
) -> str:
    command = f"""{jax_preflight(run_root, "smoke")}
"$PY" main.py \\
  --env_name=bridge-cube-single-play-singletask-task1-v0 \\
  --dataset_dir="$DATA_ROOT" \\
  --save_dir="$RUN_ROOT/smoke_runs" \\
  --wandb_run_group=bridge_state_only_infom_smoke \\
  --run_id=smoke \\
  --enable_wandb=0 \\
  --enable_tensorboard=1 \\
  --resume_from_checkpoint=1 \\
  --checkpoint_interval=2 \\
  --checkpoint_keep=3 \\
  --checkpoint_signal_exit_code={CHECKPOINT_SIGNAL_EXIT_CODE} \\
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
  --agent.alpha=30 \\
  --agent.bridge_loss_weight=0.0 &
child=$!
forward_checkpoint_signal() {{
  echo "Forwarding checkpoint signal to Python child $child"
  kill -USR1 "$child" 2>/dev/null || true
}}
trap forward_checkpoint_signal USR1 TERM
wait "$child"
"""
    return f"""#!/bin/bash
#SBATCH --job-name=bridge-state-smoke
#SBATCH --partition={partition}
{optional_sbatch("constraint", constraint)}{optional_sbatch("gres", gres)}#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output={run_root}/slurm/%x-%j.out
#SBATCH --error={run_root}/slurm/%x-%j.err

set -euo pipefail
{common_exports(repo, python, run_root, data_root)}
echo "host=$(hostname) job=${{SLURM_JOB_ID}} stage=smoke date=$(date -Is) repo=$(git rev-parse HEAD)"
git status --short --branch
nvidia-smi
{requeue_wrapper(command)}
"""


def train_script(
    repo: Path,
    python: Path,
    run_root: Path,
    data_root: Path,
    task: int,
    seed: int,
    partition: str,
    constraint: str,
    cpus: int,
    mem: str,
    gres: str,
    time_limit: str,
    checkpoint_interval: int,
    checkpoint_keep: int,
) -> str:
    label = f"train_t{task}_s{seed}"
    command = f"""{jax_preflight(run_root, label)}
"$PY" main.py \\
  --env_name=bridge-cube-single-play-singletask-task${{TASK}}-v0 \\
  --dataset_dir="$DATA_ROOT" \\
  --save_dir="$RUN_ROOT/runs" \\
  --wandb_run_group="$GROUP" \\
  --run_id="task${{TASK}}_seed${{SEED}}" \\
  --enable_wandb=0 \\
  --enable_tensorboard=1 \\
  --resume_from_checkpoint=1 \\
  --checkpoint_interval={checkpoint_interval} \\
  --checkpoint_keep={checkpoint_keep} \\
  --checkpoint_signal_exit_code={CHECKPOINT_SIGNAL_EXIT_CODE} \\
  --pretraining_steps=1000000 \\
  --finetuning_steps=500000 \\
  --eval_interval=50000 \\
  --eval_episodes=50 \\
  --seed="$SEED" \\
  --agent=agents/infom.py \\
  --agent.expectile=0.95 \\
  --agent.kl_weight=0.05 \\
  --agent.alpha=30 \\
  --agent.bridge_loss_weight=0.0 &
child=$!
forward_checkpoint_signal() {{
  echo "Forwarding checkpoint signal to Python child $child"
  kill -USR1 "$child" 2>/dev/null || true
}}
trap forward_checkpoint_signal USR1 TERM
wait "$child"
"""
    return f"""#!/bin/bash
#SBATCH --job-name=bridge-state-t{task}-s{seed}
#SBATCH --partition={partition}
{optional_sbatch("constraint", constraint)}{optional_sbatch("gres", gres)}#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time_limit}
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --output={run_root}/slurm/%x-%j.out
#SBATCH --error={run_root}/slurm/%x-%j.err

set -euo pipefail
{common_exports(repo, python, run_root, data_root)}
TASK={task}
SEED={seed}
GROUP=bridge_state_only_infom_task${{TASK}}
echo "host=$(hostname) job=${{SLURM_JOB_ID}} stage=train task=${{TASK}} seed=${{SEED}} date=$(date -Is) repo=$(git rev-parse HEAD)"
git status --short --branch
nvidia-smi
{requeue_wrapper(command)}
"""


def main() -> int:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-host", default="unity")
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=DEFAULT_BASE
        / f"infom-learning-from-obs/runs/bridge_state_only_infom_alltasks_{timestamp}",
    )
    parser.add_argument("--tasks", default="1-5")
    parser.add_argument("--seeds", default="0-1")
    parser.add_argument("--audit-job-id", default="56794980")
    parser.add_argument("--partition", default="gpu,gpu-preempt")
    parser.add_argument("--constraint", default="h100|a100|l40s|a40")
    parser.add_argument("--gres", default="gpu:1")
    parser.add_argument("--time-limit", default="04:00:00")
    parser.add_argument("--cpus-per-task", type=int, default=8)
    parser.add_argument("--mem", default="32G")
    parser.add_argument("--checkpoint-interval", type=int, default=50_000)
    parser.add_argument("--checkpoint-keep", type=int, default=3)
    parser.add_argument("--no-submit", action="store_true")
    args = parser.parse_args()

    tasks = parse_range_list(args.tasks)
    seeds = parse_range_list(args.seeds)
    run_root = args.run_root
    slurm_dir = run_root / "slurm"

    remote_run(args.ssh_host, f"mkdir -p {shlex.quote(str(slurm_dir))}")

    scripts: dict[str, str] = {"smoke": smoke_script(
        args.repo,
        args.python,
        run_root,
        args.data_root,
        args.partition,
        args.constraint,
        args.cpus_per_task,
        args.mem,
        args.gres,
    )}
    for task in tasks:
        for seed in seeds:
            scripts[f"train_t{task}_s{seed}"] = train_script(
                args.repo,
                args.python,
                run_root,
                args.data_root,
                task,
                seed,
                args.partition,
                args.constraint,
                args.cpus_per_task,
                args.mem,
                args.gres,
                args.time_limit,
                args.checkpoint_interval,
                args.checkpoint_keep,
            )

    manifest = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_root": str(run_root),
        "data_root": str(args.data_root),
        "repo": str(args.repo),
        "python": str(args.python),
        "tasks": tasks,
        "seeds": seeds,
        "audit_job_id": args.audit_job_id,
        "partition": args.partition,
        "constraint": args.constraint,
        "gres": args.gres,
        "time_limit": args.time_limit,
        "cpus_per_task": args.cpus_per_task,
        "mem": args.mem,
        "checkpoint_interval": args.checkpoint_interval,
        "checkpoint_keep": args.checkpoint_keep,
        "method": "state-only InFOM on bridge files, bridge_loss_weight=0.0",
    }

    with tempfile.TemporaryDirectory(prefix="bridge_state_submit_") as tmp:
        tmp_path = Path(tmp)
        local_slurm = tmp_path / "slurm"
        local_slurm.mkdir()
        remote_scripts: dict[str, Path] = {}
        for stage, text in scripts.items():
            local_path = local_slurm / f"{stage}.sbatch"
            local_path.write_text(text)
            local_path.chmod(0o755)
            remote_path = slurm_dir / local_path.name
            remote_copy(args.ssh_host, local_path, remote_path)
            remote_scripts[stage] = remote_path

        local_manifest = tmp_path / "submission_manifest.json"
        manifest["scripts"] = {stage: str(path) for stage, path in remote_scripts.items()}
        local_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        remote_copy(args.ssh_host, local_manifest, run_root / "submission_manifest.json")

    if args.no_submit:
        print(f"Wrote scripts under {slurm_dir}")
        return 0

    jobs: list[tuple[str, str, Path]] = []
    smoke_dependency = f"afterok:{args.audit_job_id}" if args.audit_job_id else None
    smoke_id = sbatch(args.ssh_host, remote_scripts["smoke"], smoke_dependency)
    jobs.append(("smoke", smoke_id, remote_scripts["smoke"]))
    print(f"smoke {smoke_id}")

    train_dependency = f"afterok:{smoke_id}"
    for stage in sorted(remote_scripts):
        if not re.fullmatch(r"train_t\d+_s\d+", stage):
            continue
        job_id = sbatch(args.ssh_host, remote_scripts[stage], train_dependency)
        jobs.append((stage, job_id, remote_scripts[stage]))
        print(f"{stage} {job_id}")

    local_jobs = tempfile.NamedTemporaryFile("w", delete=False, prefix="bridge_state_jobs_", suffix=".tsv")
    try:
        with local_jobs:
            local_jobs.write("stage\tjob_id\tscript\n")
            for stage, job_id, script in jobs:
                local_jobs.write(f"{stage}\t{job_id}\t{script}\n")
        remote_copy(args.ssh_host, Path(local_jobs.name), run_root / "jobs.tsv")
    finally:
        Path(local_jobs.name).unlink(missing_ok=True)

    print(f"run_root {run_root}")
    print(f"jobs {run_root / 'jobs.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
