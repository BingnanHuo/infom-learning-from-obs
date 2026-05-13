#!/usr/bin/env python3
"""Submit cube-single paired bridge data generation jobs on Unity."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


DEFAULT_BASE = Path("/work/pi_mengfanxu_umass_edu/bhuo_umass_edu")
DEFAULT_REPO = DEFAULT_BASE / "repos/infom-learning-from-obs"
DEFAULT_PYTHON = DEFAULT_BASE / "tools/miniforge3/envs/infom-obs/bin/python"


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


def sbatch(host: str, script: Path, dependency: str | None = None) -> str:
    remote_cmd = ["sbatch", "--parsable"]
    if dependency:
        remote_cmd.append(f"--dependency={dependency}")
    remote_cmd.append(str(script))
    output = remote_run(host, " ".join(remote_cmd))
    for line in reversed(output.splitlines()):
        candidate = line.split(";", 1)[0].strip()
        if re.fullmatch(r"\d+(?:_\d+)?", candidate):
            return candidate
    raise RuntimeError(f"Could not parse sbatch job id from output:\n{output}")


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)


def optional_sbatch(name: str, value: str | None) -> str:
    if value is None or value == "":
        return ""
    return f"#SBATCH --{name}={value}\n"


def common_exports(repo: Path, python: Path, run_root: Path, data_root: Path) -> str:
    return f"""RUN_ROOT={run_root}
DATA_ROOT={data_root}
REPO={repo}
PY={python}
export RUN_ROOT DATA_ROOT REPO PY
export PYTHONPATH="$REPO"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_TRACEBACK_FILTERING=off
mkdir -p "$RUN_ROOT" "$DATA_ROOT"
cd "$REPO"
echo "host=$(hostname) job=${{SLURM_JOB_ID}} date=$(date -Is)"
echo "repo=$(git rev-parse HEAD)"
git status --short --branch
"""


def generation_script(
    repo: Path,
    python: Path,
    run_root: Path,
    data_root: Path,
    code_dir: Path,
    split: str,
    chunk_idx: int,
    episodes: int,
    mem: str,
    time_limit: str,
    partition: str,
    cpus: int,
    gres: str | None,
    constraint: str | None,
    seed: int,
    render_width: int,
    render_height: int,
    render_camera: str,
) -> str:
    if split == "pretrain":
        chunk_root = f"$RUN_ROOT/chunks/pretrain/pretrain_chunk_{chunk_idx:03d}"
        job_name = f"bridge-pre-{chunk_idx:03d}"
    elif split == "finetune":
        chunk_root = f"$RUN_ROOT/chunks/finetune/finetune_chunk_{chunk_idx:03d}"
        job_name = f"bridge-ft-{chunk_idx:03d}"
    else:
        raise ValueError(split)
    output_args = f'--save_path="{chunk_root}.npz"'

    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
{optional_sbatch("gres", gres)}{optional_sbatch("constraint", constraint)}#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time_limit}
#SBATCH --output={run_root}/slurm/%x-%j.out
#SBATCH --error={run_root}/slurm/%x-%j.err

set -euo pipefail
{common_exports(repo, python, run_root, data_root)}
mkdir -p "$RUN_ROOT/chunks/{split}"
"$PY" "{code_dir}/generate_ogbench_bridge_dataset.py" \\
  --env_name=cube-single-v0 \\
  --dataset_type=play \\
  {output_args} \\
  --num_episodes={episodes} \\
  --max_episode_steps=1001 \\
  --seed={seed} \\
  --render_camera={render_camera} \\
  --render_width={render_width} \\
  --render_height={render_height}
ls -lh "$RUN_ROOT/chunks/{split}"
"""


def aggregate_script(
    repo: Path,
    python: Path,
    run_root: Path,
    data_root: Path,
    code_dir: Path,
    partition: str,
    cpus: int,
    mem: str,
    time_limit: str,
) -> str:
    return f"""#!/bin/bash
#SBATCH --job-name=bridge-data-agg
#SBATCH --partition={partition}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time_limit}
#SBATCH --output={run_root}/slurm/%x-%j.out
#SBATCH --error={run_root}/slurm/%x-%j.err

set -euo pipefail
{common_exports(repo, python, run_root, data_root)}
"$PY" "{code_dir}/aggregate_ogbench_bridge_chunks.py" \\
  --chunk-dir="$RUN_ROOT/chunks/pretrain" \\
  --output-dir="$DATA_ROOT" \\
  --prefix=bridge-cube-single-play-v0 \\
  --chunk-glob='pretrain_chunk_*.npz' \\
  --val-chunk-glob='pretrain_chunk_*-val.npz'
"$PY" "{code_dir}/aggregate_ogbench_bridge_chunks.py" \\
  --chunk-dir="$RUN_ROOT/chunks/finetune" \\
  --output-dir="$DATA_ROOT" \\
  --prefix=bridge-cube-single-play-ft-v0 \\
  --chunk-glob='finetune_chunk_*.npz' \\
  --val-chunk-glob='finetune_chunk_*-val.npz'
ls -lh "$DATA_ROOT"
"""


def audit_script(
    repo: Path,
    python: Path,
    run_root: Path,
    data_root: Path,
    code_dir: Path,
    partition: str,
    cpus: int,
    mem: str,
    time_limit: str,
) -> str:
    return f"""#!/bin/bash
#SBATCH --job-name=bridge-data-audit
#SBATCH --partition={partition}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time_limit}
#SBATCH --output={run_root}/slurm/%x-%j.out
#SBATCH --error={run_root}/slurm/%x-%j.err

set -euo pipefail
{common_exports(repo, python, run_root, data_root)}
mkdir -p "$RUN_ROOT/audit"
"$PY" "{code_dir}/audit_ogbench_bridge_data.py" \\
  --dataset-dir="$DATA_ROOT" \\
  --task-splits=both \\
  --json-out="$RUN_ROOT/audit/bridge_data_audit.json"
"""


def main() -> int:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-host", default="unity")
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=DEFAULT_BASE / f"infom-learning-from-obs/runs/unity_bridge_cube_single_data_{timestamp}",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_BASE / f"infom-learning-from-obs/data/bridge_cube_single_play_{timestamp}",
    )
    parser.add_argument("--partition", default="cpu")
    parser.add_argument("--generation-partition", default="gpu-preempt")
    parser.add_argument("--generation-gres", default="gpu:1")
    parser.add_argument("--generation-constraint", default="")
    parser.add_argument("--aggregate-partition", default="cpu")
    parser.add_argument("--audit-partition", default="cpu")
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--pretrain-mem", default="128G")
    parser.add_argument("--finetune-mem", default="96G")
    parser.add_argument("--aggregate-mem", default="160G")
    parser.add_argument("--audit-mem", default="128G")
    parser.add_argument("--pretrain-time", default="04:00:00")
    parser.add_argument("--finetune-time", default="04:00:00")
    parser.add_argument("--aggregate-time", default="06:00:00")
    parser.add_argument("--audit-time", default="04:00:00")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-stride", type=int, default=100_000)
    parser.add_argument("--pretrain-episodes", type=int, default=1000)
    parser.add_argument("--finetune-episodes", type=int, default=500)
    parser.add_argument("--pretrain-chunks", type=int, default=10)
    parser.add_argument("--finetune-chunks", type=int, default=5)
    parser.add_argument("--render-width", type=int, default=64)
    parser.add_argument("--render-height", type=int, default=64)
    parser.add_argument("--render-camera", default="front_pixels")
    parser.add_argument("--no-submit", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    local_files = [
        repo_root / "data_gen_scripts/generate_ogbench_bridge_dataset.py",
        repo_root / "scripts/aggregate_ogbench_bridge_chunks.py",
        repo_root / "scripts/audit_ogbench_bridge_data.py",
    ]

    code_dir = args.run_root / "code"
    script_dir = args.run_root / "slurm"
    manifest = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_root": str(args.run_root),
        "data_root": str(args.data_root),
        "repo": str(args.repo),
        "python": str(args.python),
        "partition": args.partition,
        "generation_partition": args.generation_partition,
        "generation_gres": args.generation_gres,
        "generation_constraint": args.generation_constraint,
        "aggregate_partition": args.aggregate_partition,
        "audit_partition": args.audit_partition,
        "seed": args.seed,
        "seed_stride": args.seed_stride,
        "pretrain_episodes": args.pretrain_episodes,
        "finetune_episodes": args.finetune_episodes,
        "pretrain_chunks": args.pretrain_chunks,
        "finetune_chunks": args.finetune_chunks,
        "render_width": args.render_width,
        "render_height": args.render_height,
        "render_camera": args.render_camera,
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        local_code_dir = tmp_path / "code"
        local_script_dir = tmp_path / "slurm"
        local_code_dir.mkdir()
        for path in local_files:
            shutil.copy2(path, local_code_dir / path.name)

        def split_counts(total: int, chunks: int) -> list[int]:
            if chunks <= 0:
                raise ValueError("chunk count must be positive")
            base = total // chunks
            rem = total % chunks
            return [base + (1 if idx < rem else 0) for idx in range(chunks)]

        scripts = {"audit": local_script_dir / "audit_bridge_data.sbatch"}
        pretrain_counts = split_counts(args.pretrain_episodes, args.pretrain_chunks)
        finetune_counts = split_counts(args.finetune_episodes, args.finetune_chunks)
        for idx, episodes in enumerate(pretrain_counts):
            key = f"pretrain_{idx:03d}"
            scripts[key] = local_script_dir / f"generate_pretrain_{idx:03d}.sbatch"
            write(
                scripts[key],
                generation_script(
                    args.repo,
                    args.python,
                    args.run_root,
                    args.data_root,
                    code_dir,
                    "pretrain",
                    idx,
                    episodes,
                    args.pretrain_mem,
                    args.pretrain_time,
                    args.generation_partition or args.partition,
                    args.cpus,
                    args.generation_gres,
                    args.generation_constraint,
                    args.seed + idx * args.seed_stride,
                    args.render_width,
                    args.render_height,
                    args.render_camera,
                ),
            )
        for idx, episodes in enumerate(finetune_counts):
            key = f"finetune_{idx:03d}"
            scripts[key] = local_script_dir / f"generate_finetune_{idx:03d}.sbatch"
            write(
                scripts[key],
                generation_script(
                    args.repo,
                    args.python,
                    args.run_root,
                    args.data_root,
                    code_dir,
                    "finetune",
                    idx,
                    episodes,
                    args.finetune_mem,
                    args.finetune_time,
                    args.generation_partition or args.partition,
                    args.cpus,
                    args.generation_gres,
                    args.generation_constraint,
                    args.seed + 1_000_000 + idx * args.seed_stride,
                    args.render_width,
                    args.render_height,
                    args.render_camera,
                ),
            )
        scripts["aggregate"] = local_script_dir / "aggregate_bridge_data.sbatch"
        write(
            scripts["aggregate"],
            aggregate_script(
                args.repo,
                args.python,
                args.run_root,
                args.data_root,
                code_dir,
                args.aggregate_partition or args.partition,
                args.cpus,
                args.aggregate_mem,
                args.aggregate_time,
            ),
        )
        write(
            scripts["audit"],
            audit_script(
                args.repo,
                args.python,
                args.run_root,
                args.data_root,
                code_dir,
                args.audit_partition or args.partition,
                args.cpus,
                args.audit_mem,
                args.audit_time,
            ),
        )
        manifest_path = tmp_path / "submission_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

        remote_run(args.ssh_host, f"mkdir -p {code_dir} {script_dir} {args.run_root}/audit")
        run(["scp", *[str(path) for path in local_code_dir.iterdir()], f"{args.ssh_host}:{code_dir}/"])
        run(["scp", *[str(path) for path in local_script_dir.iterdir()], f"{args.ssh_host}:{script_dir}/"])
        run(["scp", str(manifest_path), f"{args.ssh_host}:{args.run_root}/submission_manifest.json"])

    manifest["scripts"] = {key: str(script_dir / path.name) for key, path in scripts.items()}

    if not args.no_submit:
        generation_jobs = {}
        for key, script in manifest["scripts"].items():
            if key.startswith("pretrain_") or key.startswith("finetune_"):
                generation_jobs[key] = sbatch(args.ssh_host, Path(script))
        dependency_ids = ":".join(generation_jobs.values())
        aggregate_job = sbatch(
            args.ssh_host,
            script_dir / "aggregate_bridge_data.sbatch",
            dependency=f"afterok:{dependency_ids}",
        )
        audit_job = sbatch(
            args.ssh_host,
            script_dir / "audit_bridge_data.sbatch",
            dependency=f"afterok:{aggregate_job}",
        )
        manifest["jobs"] = {
            **generation_jobs,
            "aggregate": aggregate_job,
            "audit": audit_job,
        }
        remote_manifest = json.dumps(manifest, indent=2, sort_keys=True)
        remote_run(
            args.ssh_host,
            "cat > "
            f"{args.run_root}/submission_manifest.json <<'EOF'\n{remote_manifest}\nEOF",
        )

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
