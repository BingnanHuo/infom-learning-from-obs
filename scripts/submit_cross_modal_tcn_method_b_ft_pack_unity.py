#!/usr/bin/env python3
"""Submit one packed short-QOS Method B fine-tuning job on Unity."""

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
DEFAULT_B256_CKPT = (
    DEFAULT_BASE
    / "infom-learning-from-obs/runs/cross_modal_tcn_method_b_diag_20260506_232914"
    / "runs/cross_modal_tcn_method_b_diagnostic/task1_seed0_1000kpre_0kft/checkpoints"
)
DEFAULT_B512_CKPT = (
    DEFAULT_BASE
    / "infom-learning-from-obs/runs/cross_modal_tcn_method_b_diag_20260506_205707"
    / "runs/cross_modal_tcn_method_b_diagnostic/task1_seed0_1000kpre_0kft/checkpoints"
)
CHECKPOINT_SIGNAL_EXIT_CODE = 75
JAX_PREFLIGHT_EXIT_CODE = 76
SIGNAL_REQUEUE_EXIT_CODES = (138, 143)
DEFAULT_RUN_ORDER = "4:0,4:1,1:0,1:1,5:0,5:1,3:0,3:1,2:0,2:1"


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
    output = remote_run(host, " ".join(shlex.quote(part) for part in ["sbatch", "--parsable", str(script)]))
    for line in reversed(output.splitlines()):
        candidate = line.split(";", 1)[0].strip()
        if re.fullmatch(r"\d+(?:_\d+)?", candidate):
            return candidate
    raise RuntimeError(f"Could not parse sbatch job id from output:\n{output}")


def optional_sbatch(name: str, value: str | None) -> str:
    if value is None or value == "":
        return ""
    return f"#SBATCH --{name}={value}\n"


def parse_run_order(text: str) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    for item in text.split(","):
        task, seed = item.split(":", 1)
        runs.append((int(task), int(seed)))
    return runs


def bash_array(values: list[int]) -> str:
    return "(" + " ".join(shlex.quote(str(v)) for v in values) + ")"


def packed_script(args: argparse.Namespace, run_root: Path, runs: list[tuple[int, int]]) -> str:
    tasks = bash_array([task for task, _ in runs])
    seeds = bash_array([seed for _, seed in runs])
    short_name = f"b{args.shared_latent_dim}"
    group = args.wandb_run_group or f"cross_modal_tcn_{short_name}_ft_pack_short"
    mem_line = f"#SBATCH --mem={args.mem}\n" if args.mem else ""
    requeue_codes = " ".join(
        str(code)
        for code in (CHECKPOINT_SIGNAL_EXIT_CODE, JAX_PREFLIGHT_EXIT_CODE, *SIGNAL_REQUEUE_EXIT_CODES)
    )

    return f"""#!/bin/bash
#SBATCH --job-name=tcn-{short_name}-ft-pack
#SBATCH --partition={args.partition}
{optional_sbatch("qos", args.qos)}{optional_sbatch("dependency", args.dependency)}{optional_sbatch("constraint", args.constraint)}#SBATCH --nodes=1
#SBATCH --gres=gpu:{args.gpus}
#SBATCH --cpus-per-task={args.cpus}
{mem_line}#SBATCH --time={args.time_limit}
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --output={run_root}/slurm/%x-%j.out
#SBATCH --error={run_root}/slurm/%x-%j.err

set -euo pipefail

RUN_ROOT={run_root}
REPO={args.repo}
PY={args.python}
DATA_ROOT={args.data_root}
PRETRAIN_CKPT={args.resume_checkpoint_path}
GROUP={group}
LATENT_DIM={args.shared_latent_dim}
GPU_COUNT={args.gpus}
export RUN_ROOT REPO PY DATA_ROOT PRETRAIN_CKPT GROUP LATENT_DIM GPU_COUNT
export PYTHONPATH="$REPO"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_TRACEBACK_FILTERING=off

mkdir -p "$RUN_ROOT" "$RUN_ROOT/probe_results" "$RUN_ROOT/slurm" "$RUN_ROOT/child_status"
cd "$REPO"
echo "host=$(hostname) job=${{SLURM_JOB_ID}} qos={args.qos} latent_dim=$LATENT_DIM date=$(date -Is) repo=$(git rev-parse HEAD)"
git status --short --branch
nvidia-smi

TASKS={tasks}
SEEDS={seeds}

if [ -n "${{CUDA_VISIBLE_DEVICES:-}}" ]; then
  IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
else
  GPU_IDS=()
  for ((g=0; g<GPU_COUNT; g++)); do GPU_IDS+=("$g"); done
fi

if [ "${{#GPU_IDS[@]}}" -lt "$GPU_COUNT" ]; then
  echo "Expected $GPU_COUNT visible GPUs, found ${{#GPU_IDS[@]}}: ${{CUDA_VISIBLE_DEVICES:-unset}}" >&2
  exit {JAX_PREFLIGHT_EXIT_CODE}
fi

children=()
requeue_needed=0
failed=0
REQUEUE_CODES="{requeue_codes}"

forward_checkpoint_signal() {{
  echo "Forwarding checkpoint signal to ${{#children[@]}} children"
  for child in "${{children[@]:-}}"; do
    kill -USR1 "$child" 2>/dev/null || true
  done
}}
trap forward_checkpoint_signal USR1 TERM

run_one() {{
  local idx="$1"
  local gpu="$2"
  local task="${{TASKS[$idx]}}"
  local seed="${{SEEDS[$idx]}}"
  local run_id="task${{task}}_seed${{seed}}_1000kpre_500kft_from_b${{LATENT_DIM}}_1m_shortpack"
  local log="$RUN_ROOT/slurm/${{run_id}}-${{SLURM_JOB_ID}}.log"
  local py_child=""
  forward_child_checkpoint_signal() {{
    if [ -n "$py_child" ]; then
      echo "Forwarding checkpoint signal to $run_id Python child $py_child"
      kill -USR1 "$py_child" 2>/dev/null || true
    fi
  }}
  trap forward_child_checkpoint_signal USR1 TERM
  {{
    set +e
    echo "Starting $run_id on CUDA_VISIBLE_DEVICES=$gpu at $(date -Is)"
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PY" scripts/jax_cuda_probe.py \\
      --output="$RUN_ROOT/probe_results/${{run_id}}_${{SLURM_JOB_ID}}.json"
    local code=$?
    if [ "$code" != "0" ]; then
      echo "JAX CUDA preflight failed for $run_id with code $code"
      echo "{JAX_PREFLIGHT_EXIT_CODE}" > "$RUN_ROOT/child_status/${{run_id}}.status"
      exit {JAX_PREFLIGHT_EXIT_CODE}
    fi
    "$PY" main.py \\
      --env_name=bridge-cube-single-play-singletask-task${{task}}-v0 \\
      --dataset_dir="$DATA_ROOT" \\
      --save_dir="$RUN_ROOT/runs" \\
      --wandb_run_group="$GROUP" \\
      --run_id="$run_id" \\
      --enable_wandb=0 \\
      --enable_tensorboard=1 \\
      --resume_from_checkpoint=1 \\
      --resume_checkpoint_path="$PRETRAIN_CKPT" \\
      --checkpoint_at_end=1 \\
      --checkpoint_interval={args.checkpoint_interval} \\
      --checkpoint_keep={args.checkpoint_keep} \\
      --checkpoint_signal_exit_code={CHECKPOINT_SIGNAL_EXIT_CODE} \\
      --pretraining_steps={args.pretraining_steps} \\
      --pretraining_size={args.pretraining_size} \\
      --finetuning_steps={args.finetuning_steps} \\
      --finetuning_size={args.finetuning_size} \\
      --log_interval={args.log_interval} \\
      --eval_interval={args.eval_interval} \\
      --eval_episodes={args.eval_episodes} \\
      --save_interval=999999999 \\
      --save_best_eval=1 \\
      --best_eval_metric=evaluation/success \\
      --obs_norm_type=none \\
      --seed="${{seed}}" \\
      --agent=agents/cross_modal_tcn_infom.py \\
      --agent.batch_size={args.batch_size} \\
      --agent.expectile=0.95 \\
      --agent.kl_weight=0.05 \\
      --agent.alpha=30 \\
      --agent.rgb_encoder=impala_small \\
      --agent.shared_latent_dim="$LATENT_DIM" \\
      --agent.infonce_temperature={args.infonce_temperature} \\
      --agent.lambda_infonce={args.lambda_infonce} \\
      --agent.lambda_bc_state_extra={args.lambda_bc_state_extra} \\
      --agent.device_bridge_cache=1 \\
      --agent.warmup_align_steps={args.warmup_align_steps} &
    py_child=$!
    wait "$py_child"
    code=$?
  }} >> "$log" 2>&1
  echo "$code" > "$RUN_ROOT/child_status/${{run_id}}.status"
  echo "Finished $run_id with code $code at $(date -Is)" | tee -a "$log"
  exit "$code"
}}

for ((start=0; start<${{#TASKS[@]}}; start+=GPU_COUNT)); do
  children=()
  for ((slot=0; slot<GPU_COUNT; slot++)); do
    idx=$((start + slot))
    if [ "$idx" -ge "${{#TASKS[@]}}" ]; then
      break
    fi
    run_one "$idx" "${{GPU_IDS[$slot]}}" &
    children+=("$!")
  done

  for child in "${{children[@]}}"; do
    set +e
    wait "$child"
    code=$?
    set -e
    if [[ " $REQUEUE_CODES " == *" $code "* ]]; then
      requeue_needed=1
    elif [ "$code" != "0" ]; then
      failed=1
    fi
  done

  if [ "$requeue_needed" = "1" ] || [ "$failed" = "1" ]; then
    break
  fi
done

if [ "$requeue_needed" = "1" ]; then
  echo "Requeueing ${{SLURM_JOB_ID}} after checkpoint/preflight child exit"
  scontrol requeue "${{SLURM_JOB_ID}}"
  exit 0
fi

if [ "$failed" = "1" ]; then
  echo "At least one packed child failed; see $RUN_ROOT/slurm/*.log" >&2
  exit 1
fi

echo "Packed Method B FT job finished at $(date -Is)"
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-host", default="bhuo_umass_edu@unity")
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--resume-checkpoint-path", type=Path, default=None)
    parser.add_argument("--remote-run-root", type=Path, default=None)
    parser.add_argument("--local-run-root", type=Path, default=Path("runs/unity_cross_modal_tcn_method_b"))
    parser.add_argument("--partition", default="gpu,gpu-preempt")
    parser.add_argument("--qos", default="short")
    parser.add_argument("--dependency", default=None)
    parser.add_argument("--constraint", default="h100|a100-80g|l40s|a100-40g")
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--cpus", type=int, default=32)
    parser.add_argument("--mem", default="480G")
    parser.add_argument("--time-limit", default="03:50:00")
    parser.add_argument("--shared-latent-dim", type=int, choices=[256, 512], default=256)
    parser.add_argument("--run-order", default=DEFAULT_RUN_ORDER)
    parser.add_argument("--wandb-run-group", default=None)
    parser.add_argument("--pretraining-steps", type=int, default=1_000_000)
    parser.add_argument("--pretraining-size", type=int, default=1_000_000)
    parser.add_argument("--finetuning-steps", type=int, default=500_000)
    parser.add_argument("--finetuning-size", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--checkpoint-interval", type=int, default=50_000)
    parser.add_argument("--checkpoint-keep", type=int, default=5)
    parser.add_argument("--warmup-align-steps", type=int, default=10_000)
    parser.add_argument("--infonce-temperature", type=float, default=0.1)
    parser.add_argument("--lambda-infonce", type=float, default=1.0)
    parser.add_argument("--lambda-bc-state-extra", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.resume_checkpoint_path is None:
        args.resume_checkpoint_path = (
            DEFAULT_B256_CKPT if args.shared_latent_dim == 256 else DEFAULT_B512_CKPT
        )

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    short_name = f"b{args.shared_latent_dim}"
    run_root = args.remote_run_root or (
        DEFAULT_BASE / f"infom-learning-from-obs/runs/cross_modal_tcn_{short_name}_ft_pack_short_{timestamp}"
    )
    local_root = args.local_run_root / f"cross_modal_tcn_{short_name}_ft_pack_short_{timestamp}"
    local_slurm = local_root / "slurm"
    local_slurm.mkdir(parents=True, exist_ok=True)

    runs = parse_run_order(args.run_order)
    local_script = local_slurm / f"{short_name}_ft_pack.sbatch"
    local_script.write_text(packed_script(args, run_root, runs))

    manifest = {
        "created_at": timestamp,
        "remote_run_root": str(run_root),
        "local_run_root": str(local_root),
        "remote_repo": str(args.repo),
        "data_root": str(args.data_root),
        "resume_checkpoint_path": str(args.resume_checkpoint_path),
        "partition": args.partition,
        "qos": args.qos,
        "dependency": args.dependency,
        "constraint": args.constraint,
        "gpus": args.gpus,
        "cpus": args.cpus,
        "mem": args.mem,
        "time_limit": args.time_limit,
        "shared_latent_dim": args.shared_latent_dim,
        "run_order": [{"task": task, "seed": seed} for task, seed in runs],
        "wandb_run_group": args.wandb_run_group or f"cross_modal_tcn_{short_name}_ft_pack_short",
        "pretraining_steps": args.pretraining_steps,
        "finetuning_steps": args.finetuning_steps,
        "eval_interval": args.eval_interval,
        "eval_episodes": args.eval_episodes,
        "checkpoint_interval": args.checkpoint_interval,
        "checkpoint_keep": args.checkpoint_keep,
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
