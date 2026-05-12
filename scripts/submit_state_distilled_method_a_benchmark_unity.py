#!/usr/bin/env python3
"""Submit A100/H100 State-Distilled InFOM performance benchmarks on Unity."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shlex
import subprocess
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


def sbatch(host: str, script: Path) -> str:
    cmd = ["sbatch", "--parsable", str(script)]
    output = remote_run(host, " ".join(shlex.quote(part) for part in cmd))
    for line in reversed(output.splitlines()):
        candidate = line.split(";", 1)[0].strip()
        if re.fullmatch(r"\d+(?:_\d+)?", candidate):
            return candidate
    raise RuntimeError(f"Could not parse sbatch job id from output:\n{output}")


def optional_sbatch(name: str, value: str | None) -> str:
    if value is None or value == "":
        return ""
    return f"#SBATCH --{name}={value}\n"


def jax_preflight(run_root: Path) -> str:
    return f""""$PY" scripts/jax_cuda_probe.py \\
  --output="{run_root}/probe_results/method_a_perf_${{SLURM_JOB_ID}}.json"
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


def smoke_command(args: argparse.Namespace, p_aug: float, label: str, device_cache: bool) -> str:
    device_cache_arg = "  --agent.device_bridge_cache=1 \\\n" if device_cache else ""
    return f""""$PY" main.py \\
  --env_name=bridge-cube-single-play-singletask-task{args.task}-v0 \\
  --dataset_dir="$DATA_ROOT" \\
  --save_dir="$RUN_ROOT/runs" \\
  --wandb_run_group=state_distilled_method_a_perf_{label} \\
  --run_id=task{args.task}_seed{args.seed}_{args.smoke_steps}step_{label} \\
  --enable_wandb=0 \\
  --enable_tensorboard=1 \\
  --resume_from_checkpoint=1 \\
  --checkpoint_interval={args.checkpoint_interval} \\
  --checkpoint_keep=2 \\
  --checkpoint_signal_exit_code={CHECKPOINT_SIGNAL_EXIT_CODE} \\
  --pretraining_steps={args.smoke_steps} \\
  --finetuning_steps=0 \\
  --pretraining_size={args.pretraining_size} \\
  --finetuning_size={args.finetuning_size} \\
  --log_interval={args.log_interval} \\
  --eval_interval=0 \\
  --eval_episodes=1 \\
  --save_interval=999999999 \\
  --obs_norm_type=none \\
  --seed={args.seed} \\
  --agent=agents/cross_modal_state_distilled_infom.py \\
  --agent.expectile=0.95 \\
  --agent.kl_weight=0.05 \\
  --agent.alpha=30 \\
  --agent.rgb_encoder=impala_small \\
  --agent.rgb_p_aug={p_aug} \\
{device_cache_arg}  --agent.warmup_align_steps={args.warmup_align_steps} \\
  --agent.lambda_align=1.0 \\
  --agent.lambda_bc_state_extra=1.0
"""


def slurm_script(args: argparse.Namespace, run_root: Path) -> str:
    command = f"""{jax_preflight(run_root)}
"$PY" scripts/benchmark_state_distilled_method_a_pipeline.py \\
  --env-name=bridge-cube-single-play-singletask-task{args.task}-v0 \\
  --dataset-dir="$DATA_ROOT" \\
  --output="$RUN_ROOT/benchmark/method_a_pipeline.json" \\
  --seed={args.seed} \\
  --pretraining-size={args.pretraining_size} \\
  --finetuning-size={args.finetuning_size} \\
  --batch-size={args.batch_size} \\
  --sample-batches={args.sample_batches} \\
  --update-steps={args.update_steps} \\
  --aug-probs="{args.aug_probs}" \\
  --rgb-frame-stack=3 \\
  --rgb-encoder=impala_small \\
  --warmup-align-steps={args.warmup_align_steps}

{smoke_command(args, 0.0, "aug0_cpu", device_cache=False)}
{smoke_command(args, 0.0, "aug0_device", device_cache=True)}
{smoke_command(args, args.default_aug_prob, "augdefault_device", device_cache=True)}
"""
    return f"""#!/bin/bash
#SBATCH --job-name=sd-infom-perf
#SBATCH --partition={args.partition}
{optional_sbatch("qos", args.qos)}{optional_sbatch("constraint", args.constraint)}{optional_sbatch("gres", args.gres)}#SBATCH --cpus-per-task={args.cpus}
#SBATCH --mem={args.mem}
#SBATCH --time={args.time_limit}
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --output={run_root}/slurm/%x-%j.out
#SBATCH --error={run_root}/slurm/%x-%j.err

set -euo pipefail
RUN_ROOT={run_root}
REPO={args.repo}
PY={args.python}
DATA_ROOT={args.data_root}
export RUN_ROOT REPO PY DATA_ROOT
export PYTHONPATH="$REPO"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_TRACEBACK_FILTERING=off
mkdir -p "$RUN_ROOT" "$RUN_ROOT/probe_results" "$RUN_ROOT/benchmark"
cd "$REPO"
echo "host=$(hostname) job=${{SLURM_JOB_ID}} qos={args.qos} date=$(date -Is) repo=$(git rev-parse HEAD)"
git status --short --branch
nvidia-smi
{requeue_wrapper(command)}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-host", default="bhuo_umass_edu@unity")
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--remote-run-root", type=Path, default=None)
    parser.add_argument("--local-run-root", type=Path, default=Path("runs/unity_state_distilled_method_a_perf"))
    parser.add_argument("--partition", default="gpu,gpu-preempt")
    parser.add_argument("--qos", default="short")
    parser.add_argument("--constraint", default="h100|a100")
    parser.add_argument("--gres", default="gpu:1")
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--mem", default="128G")
    parser.add_argument("--time-limit", default="03:50:00")
    parser.add_argument("--task", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pretraining-size", type=int, default=250_000)
    parser.add_argument("--finetuning-size", type=int, default=125_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sample-batches", type=int, default=50)
    parser.add_argument("--update-steps", type=int, default=20)
    parser.add_argument("--aug-probs", default="0.0,0.5,1.0")
    parser.add_argument("--default-aug-prob", type=float, default=0.5)
    parser.add_argument("--smoke-steps", type=int, default=5_000)
    parser.add_argument("--log-interval", type=int, default=1_000)
    parser.add_argument("--checkpoint-interval", type=int, default=2_500)
    parser.add_argument("--warmup-align-steps", type=int, default=10_000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = args.remote_run_root or (
        DEFAULT_BASE / f"infom-learning-from-obs/runs/state_distilled_method_a_perf_{timestamp}"
    )
    local_root = args.local_run_root / f"state_distilled_method_a_perf_{timestamp}"
    local_slurm = local_root / "slurm"
    local_slurm.mkdir(parents=True, exist_ok=True)
    local_script = local_slurm / "method_a_perf.sbatch"
    local_script.write_text(slurm_script(args, run_root))

    manifest = {
        "created_at": timestamp,
        "remote_run_root": str(run_root),
        "local_run_root": str(local_root),
        "remote_repo": str(args.repo),
        "data_root": str(args.data_root),
        "partition": args.partition,
        "qos": args.qos,
        "constraint": args.constraint,
        "gres": args.gres,
        "time_limit": args.time_limit,
        "task": args.task,
        "seed": args.seed,
        "pretraining_size": args.pretraining_size,
        "finetuning_size": args.finetuning_size,
        "batch_size": args.batch_size,
        "smoke_steps": args.smoke_steps,
        "aug_probs": args.aug_probs,
        "default_aug_prob": args.default_aug_prob,
    }
    (local_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    remote_slurm = run_root / "slurm"
    remote_run(args.ssh_host, f"mkdir -p {shlex.quote(str(remote_slurm))}")
    remote_script = remote_slurm / local_script.name
    remote_copy(args.ssh_host, local_script, remote_script)

    if args.dry_run:
        print(json.dumps({**manifest, "remote_script": str(remote_script)}, indent=2, sort_keys=True))
        return 0

    job_id = sbatch(args.ssh_host, remote_script)
    manifest["job_id"] = job_id
    manifest["remote_script"] = str(remote_script)
    (local_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
