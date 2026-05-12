#!/usr/bin/env python3
"""Submit a small Unity job to render checkpoint comparison demos."""

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
DEFAULT_A_ROOT = (
    DEFAULT_BASE
    / "infom-learning-from-obs/runs/state_distilled_method_a_ft_grid_20260507_004541"
    / "runs/state_distilled_method_a_ft_grid"
)


def run(cmd: list[str]) -> str:
    proc = subprocess.run(
        cmd,
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


def demo_commands(args: argparse.Namespace, out_root: Path) -> list[str]:
    specs = [
        (
            "task4_seed1_best",
            "bridge-cube-single-play-singletask-task4-v0",
            DEFAULT_A_ROOT / "task4_seed1_1000kpre_500kft_from_method_a_1m/params_1350000.pkl",
            1,
        ),
        (
            "task4_seed1_final",
            "bridge-cube-single-play-singletask-task4-v0",
            DEFAULT_A_ROOT / "task4_seed1_1000kpre_500kft_from_method_a_1m/checkpoints/latest.pkl",
            1,
        ),
        (
            "task1_seed1_best",
            "bridge-cube-single-play-singletask-task1-v0",
            DEFAULT_A_ROOT / "task1_seed1_1000kpre_500kft_from_method_a_1m/params_1050000.pkl",
            1,
        ),
        (
            "task1_seed1_final",
            "bridge-cube-single-play-singletask-task1-v0",
            DEFAULT_A_ROOT / "task1_seed1_1000kpre_500kft_from_method_a_1m/checkpoints/latest.pkl",
            1,
        ),
    ]
    commands = []
    for name, env_name, checkpoint, seed in specs:
        out_dir = out_root / name
        commands.append(
            " ".join(
                shlex.quote(str(part))
                for part in [
                    args.python,
                    "scripts/render_ogbench_checkpoint_demo.py",
                    "--env-name",
                    env_name,
                    "--dataset-dir",
                    args.data_root,
                    "--agent",
                    "agents/cross_modal_state_distilled_infom.py",
                    "--checkpoint",
                    checkpoint,
                    "--out-dir",
                    out_dir,
                    "--seed",
                    seed,
                    "--eval-episodes",
                    args.eval_episodes,
                    "--video-episodes",
                    args.video_episodes,
                    "--video-frame-skip",
                    args.video_frame_skip,
                ]
            )
        )
    return commands


def slurm_script(args: argparse.Namespace, run_root: Path) -> str:
    out_root = run_root / "demos"
    commands = demo_commands(args, out_root)
    body_lines = []
    for i, command in enumerate(commands):
        body_lines.extend(
            [
                f"echo '--- demo {i + 1}/{len(commands)} start $(date -Is) ---'",
                command,
                f"echo '--- demo {i + 1}/{len(commands)} done $(date -Is) ---'",
            ]
        )
    body = "\n".join(body_lines)
    return f"""#!/bin/bash
#SBATCH --job-name=ogbench-demos
#SBATCH --partition={args.partition}
{optional_sbatch("qos", args.qos)}{optional_sbatch("constraint", args.constraint)}#SBATCH --gres={args.gres}
#SBATCH --cpus-per-task={args.cpus}
#SBATCH --mem={args.mem}
#SBATCH --time={args.time_limit}
#SBATCH --output={run_root}/slurm/%x-%j.out
#SBATCH --error={run_root}/slurm/%x-%j.err

set -euo pipefail
REPO={args.repo}
export PYTHONPATH="$REPO"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p {run_root}/slurm {out_root}
cd "$REPO"
echo "host=$(hostname) job=${{SLURM_JOB_ID}} date=$(date -Is) repo=$(git rev-parse HEAD)"
git status --short --branch
nvidia-smi
{body}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-host", default="bhuo_umass_edu@unity")
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--remote-run-root", type=Path, default=None)
    parser.add_argument("--local-run-root", type=Path, default=Path("runs/unity_checkpoint_demos"))
    parser.add_argument("--partition", default="gpu,gpu-preempt")
    parser.add_argument("--qos", default="")
    parser.add_argument("--constraint", default="h100|a100|l40s|a40")
    parser.add_argument("--gres", default="gpu:1")
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--mem", default="96G")
    parser.add_argument("--time-limit", default="01:30:00")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--video-episodes", type=int, default=2)
    parser.add_argument("--video-frame-skip", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = args.remote_run_root or (
        DEFAULT_BASE / f"infom-learning-from-obs/runs/ogbench_checkpoint_demos_{timestamp}"
    )
    local_root = args.local_run_root / f"ogbench_checkpoint_demos_{timestamp}"
    local_slurm = local_root / "slurm"
    local_slurm.mkdir(parents=True, exist_ok=True)
    local_script = local_slurm / "checkpoint_demos.sbatch"
    local_script.write_text(slurm_script(args, run_root))
    manifest = {
        "created_at": timestamp,
        "remote_run_root": str(run_root),
        "local_run_root": str(local_root),
        "partition": args.partition,
        "qos": args.qos,
        "constraint": args.constraint,
        "gres": args.gres,
        "time_limit": args.time_limit,
        "eval_episodes": args.eval_episodes,
        "video_episodes": args.video_episodes,
    }
    (local_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    remote_slurm = run_root / "slurm"
    remote_run(args.ssh_host, f"mkdir -p {shlex.quote(str(remote_slurm))}")
    remote_copy(args.ssh_host, Path("scripts/render_ogbench_checkpoint_demo.py"), args.repo / "scripts/render_ogbench_checkpoint_demo.py")
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
