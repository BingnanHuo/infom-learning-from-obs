#!/usr/bin/env python3
"""Diagnose OGBench cube-single InFOM paper-reproduction artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


EVAL_RETURN = 'evaluation/episode.return'
EVAL_SUCCESS = 'evaluation/episode.success'
TARGET_SUCCESS_PCT = {
    1: (92.5, 4.0),
    2: (78.4, 12.3),
    3: (56.4, 36.9),
    4: (91.5, 14.2),
    5: (70.0, 39.1),
}
PACKAGE_NAMES = [
    'jax',
    'jaxlib',
    'flax',
    'optax',
    'ogbench',
    'gymnasium',
    'numpy',
    'distrax',
    'ml_collections',
    'mujoco',
    'dm_control',
]
DATASET_NAMES = [
    'cube-single-play-v0.npz',
    'cube-single-play-v0-val.npz',
    'cube-single-play-ft-v0.npz',
    'cube-single-play-ft-v0-val.npz',
]
TRAIN_METRICS = [
    'training/bc/mse',
    'training/flow_occupancy/neg_elbo_loss',
    'training/flow_occupancy/kl_loss',
    'training/grad/norm',
    'training/actor/mse',
    'training/actor/q_mean',
    'training/critic/critic_loss',
    'training/reward/reward_loss',
    'validation/flow_occupancy/neg_elbo_loss',
    'validation/actor/mse',
    'validation/critic/critic_loss',
    'validation/reward/reward_loss',
]
ERROR_PATTERN = re.compile(
    r'Traceback|RuntimeError|OutOfMemory|out of memory|\bOOM\b|CANCELLED|'
    r'TIMEOUT|DUE TO TIME|FAILED|\bERROR\b|(^|\s)Error(:|\s)'
)
HOST_PATTERN = re.compile(
    r'host=(?P<host>\S+)\s+job=(?P<job>\S+)(?:\s+task=(?P<task>\S+))?'
    r'(?:\s+seed=(?P<seed>\S+))?.*repo=(?P<repo>[0-9a-f]{7,40})'
)
GPU_PATTERN = re.compile(
    r'NVIDIA-SMI\s+(?P<nvsmi>\S+)\s+Driver Version:\s+(?P<driver>\S+)'
    r'\s+CUDA Version:\s+(?P<cuda>\S+)'
)


def parse_labeled_path(value: str) -> tuple[str, Path]:
    if '=' in value:
        label, raw_path = value.split('=', 1)
        return label, Path(raw_path).expanduser().resolve()
    path = Path(value).expanduser().resolve()
    return path.name, path


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def to_float(value: str | int | float | None) -> float | None:
    if value in (None, ''):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return 'NA'
    return f'{value:.{digits}f}'


def mean_sd(values: list[float | None]) -> tuple[float | None, float | None]:
    clean = [v for v in values if v is not None]
    if not clean:
        return None, None
    if len(clean) == 1:
        return clean[0], 0.0
    return statistics.mean(clean), statistics.stdev(clean)


def run_command(cmd: list[str], cwd: Path | None = None) -> str | None:
    try:
        return subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def infer_task(flags: dict[str, Any], run_dir: Path) -> int | None:
    env_name = str(flags.get('env_name') or run_dir)
    match = re.search(r'cube-single-play-singletask-task(\d+)-v0', env_name)
    return int(match.group(1)) if match else None


def infer_seed(flags: dict[str, Any], run_dir: Path) -> int | None:
    if flags.get('seed') is not None:
        return int(flags['seed'])
    match = re.search(r'sd(\d+)', run_dir.name)
    return int(match.group(1)) if match else None


def metric_stats(rows: list[dict[str, str]], key: str) -> dict[str, Any] | None:
    values = [to_float(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return None
    finite = [value for value in values if math.isfinite(value)]
    return {
        'count': len(values),
        'nonfinite': len(values) - len(finite),
        'first': finite[0] if finite else None,
        'last': finite[-1] if finite else None,
        'min': min(finite) if finite else None,
        'max': max(finite) if finite else None,
    }


def parse_run(run_dir: Path, root_label: str) -> dict[str, Any] | None:
    eval_path = run_dir / 'finetuning_eval.csv'
    if not eval_path.exists():
        return None
    eval_rows = read_csv_rows(eval_path)
    if not eval_rows:
        return None

    flags = read_json(run_dir / 'flags.json')
    agent = flags.get('agent') or {}
    pretraining_steps = int(flags.get('pretraining_steps', 1_000_000))
    paper_steps = [
        pretraining_steps + 400_000,
        pretraining_steps + 450_000,
        pretraining_steps + 500_000,
    ]
    by_step = {int(float(row['step'])): row for row in eval_rows if row.get('step')}
    paper_successes = [to_float(by_step.get(step, {}).get(EVAL_SUCCESS)) for step in paper_steps]
    paper_returns = [to_float(by_step.get(step, {}).get(EVAL_RETURN)) for step in paper_steps]
    final = eval_rows[-1]
    pre_rows = read_csv_rows(run_dir / 'pretraining_train.csv')
    fine_rows = read_csv_rows(run_dir / 'finetuning_train.csv')
    metric_sources = pre_rows + fine_rows
    metrics = {
        metric: stat
        for metric in TRAIN_METRICS
        if (stat := metric_stats(metric_sources, metric)) is not None
    }

    success_curve = []
    return_curve = []
    for row in eval_rows:
        step = int(float(row['step']))
        success_curve.append([step, to_float(row.get(EVAL_SUCCESS))])
        return_curve.append([step, to_float(row.get(EVAL_RETURN))])

    record = {
        'root_label': root_label,
        'run_dir': str(run_dir),
        'env_name': flags.get('env_name'),
        'task': infer_task(flags, run_dir),
        'seed': infer_seed(flags, run_dir),
        'pretraining_steps': pretraining_steps,
        'finetuning_steps': int(flags.get('finetuning_steps', 500_000)),
        'eval_interval': int(flags.get('eval_interval', 50_000)),
        'eval_episodes': int(flags.get('eval_episodes', 50)),
        'pretraining_size': int(flags.get('pretraining_size', 1_000_000)),
        'finetuning_size': int(flags.get('finetuning_size', 500_000)),
        'enable_tensorboard': flags.get('enable_tensorboard'),
        'checkpoint_interval': flags.get('checkpoint_interval'),
        'checkpoint_at_end': flags.get('checkpoint_at_end'),
        'resume_from_checkpoint': flags.get('resume_from_checkpoint'),
        'dataset_dir': flags.get('dataset_dir'),
        'agent': {
            key: agent.get(key)
            for key in [
                'agent_name',
                'lr',
                'batch_size',
                'latent_dim',
                'expectile',
                'kl_weight',
                'alpha',
                'discount',
                'tau',
                'actor_freq',
                'num_flow_goals',
                'num_flow_steps',
                'critic_latent_type',
                'q_agg',
                'bridge_loss_weight',
                'encoder',
            ]
        },
        'eval_rows': len(eval_rows),
        'final_step': int(float(final['step'])),
        'final_success': to_float(final.get(EVAL_SUCCESS)),
        'final_return': to_float(final.get(EVAL_RETURN)),
        'paper_steps': paper_steps,
        'paper_success': (
            statistics.mean([v for v in paper_successes if v is not None])
            if all(v is not None for v in paper_successes)
            else None
        ),
        'paper_return': (
            statistics.mean([v for v in paper_returns if v is not None])
            if all(v is not None for v in paper_returns)
            else None
        ),
        'missing_paper_steps': [
            step for step, value in zip(paper_steps, paper_successes) if value is None
        ],
        'best_success': max(value for _, value in success_curve if value is not None),
        'success_curve': success_curve,
        'return_curve': return_curve,
        'metrics': metrics,
    }
    record['protocol_issues'] = protocol_issues(record)
    record['health_flags'] = health_flags(record)
    return record


def protocol_issues(record: dict[str, Any]) -> list[str]:
    issues = []
    checks = {
        'pretraining_steps': 1_000_000,
        'finetuning_steps': 500_000,
        'pretraining_size': 1_000_000,
        'finetuning_size': 500_000,
        'eval_interval': 50_000,
        'eval_episodes': 50,
    }
    for key, expected in checks.items():
        if record.get(key) != expected:
            issues.append(f'{key}={record.get(key)} expected {expected}')
    agent_checks = {
        'expectile': 0.95,
        'kl_weight': 0.05,
        'alpha': 30,
        'latent_dim': 512,
        'batch_size': 256,
        'lr': 0.0003,
    }
    for key, expected in agent_checks.items():
        value = record['agent'].get(key)
        if value is None or abs(float(value) - expected) > 1e-12:
            issues.append(f'agent.{key}={value} expected {expected}')
    if record['agent'].get('bridge_loss_weight') not in (None, 0, 0.0):
        issues.append(f"agent.bridge_loss_weight={record['agent'].get('bridge_loss_weight')} expected 0")
    return issues


def health_flags(record: dict[str, Any]) -> list[str]:
    flags = []
    metrics = record['metrics']

    def max_metric(key: str) -> float | None:
        stat = metrics.get(key)
        return None if stat is None else stat.get('max')

    if (value := max_metric('training/grad/norm')) is not None and value > 1_000:
        flags.append(f'training grad norm spike {value:.3g}')
    if (value := max_metric('validation/critic/critic_loss')) is not None and value > 1_000_000:
        flags.append(f'validation critic loss spike {value:.3g}')
    if (value := max_metric('validation/flow_occupancy/neg_elbo_loss')) is not None and value > 100_000:
        flags.append(f'validation flow loss spike {value:.3g}')
    if (value := max_metric('training/actor/mse')) is not None and value > 0.1:
        flags.append(f'training actor MSE spike {value:.3g}')
    q_stat = metrics.get('training/actor/q_mean')
    if q_stat is not None and q_stat.get('max') is not None and q_stat['max'] > 100:
        flags.append(f"training actor q_mean positive spike {q_stat['max']:.3g}")
    curve = [value for _, value in record['success_curve'] if value is not None]
    if curve and max(curve) - curve[-1] >= 0.5:
        flags.append(f'success collapse best {max(curve):.2f} to final {curve[-1]:.2f}')
    if record['paper_success'] is not None and record['task'] in TARGET_SUCCESS_PCT:
        target, _ = TARGET_SUCCESS_PCT[record['task']]
        if record['paper_success'] * 100 < target - 25:
            flags.append(f'paper-window success far below target by {target - record["paper_success"] * 100:.1f} pp')
    return flags


def find_run_dirs(root: Path) -> list[Path]:
    return sorted(path.parent for path in root.rglob('finetuning_eval.csv'))


def collect_root(label: str, root: Path) -> dict[str, Any]:
    records = [parse_run(run_dir, label) for run_dir in find_run_dirs(root)]
    records = [record for record in records if record is not None]
    return {
        'label': label,
        'path': str(root),
        'records': records,
        'slurm': collect_slurm(root / 'slurm'),
        'dataset_manifest': read_json(root / 'dataset_manifest.json'),
        'submission_manifest': read_json(root / 'submission_manifest.json'),
    }


def collect_slurm(slurm_dir: Path) -> dict[str, Any]:
    logs = []
    error_hits = []
    if not slurm_dir.exists():
        return {'logs': logs, 'error_hits': error_hits}
    for path in sorted(slurm_dir.glob('*')):
        if not path.is_file():
            continue
        text = path.read_text(errors='replace')
        lines = text.splitlines()
        entry = {'path': str(path), 'host': None, 'job': None, 'task': None, 'seed': None, 'repo': None, 'gpu': None}
        for line in lines[:80]:
            if match := HOST_PATTERN.search(line):
                entry.update(match.groupdict())
            if match := GPU_PATTERN.search(line):
                entry['gpu'] = match.groupdict()
        if any(value is not None for key, value in entry.items() if key != 'path'):
            logs.append(entry)
        for line_no, line in enumerate(lines, start=1):
            if ERROR_PATTERN.search(line):
                error_hits.append({'path': str(path), 'line': line_no, 'text': line[:240]})
    return {'logs': logs, 'error_hits': error_hits}


def dataset_hashes(dataset_dir: Path) -> dict[str, Any]:
    result = {'dataset_dir': str(dataset_dir), 'files': {}}
    for name in DATASET_NAMES:
        path = dataset_dir / name
        if not path.exists():
            result['files'][name] = {'missing': True}
            continue
        digest = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                digest.update(chunk)
        result['files'][name] = {
            'bytes': path.stat().st_size,
            'sha256': digest.hexdigest(),
        }
    return result


def collect_environment(repo_path: Path, dataset_dir: Path) -> dict[str, Any]:
    packages = {}
    for name in PACKAGE_NAMES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        'python': sys.version,
        'repo_path': str(repo_path),
        'git_head': run_command(['git', 'rev-parse', 'HEAD'], cwd=repo_path),
        'git_status': run_command(['git', 'status', '--short', '--branch'], cwd=repo_path),
        'packages': packages,
        'dataset_hashes': dataset_hashes(dataset_dir),
    }


def load_records(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def combine_collections(collections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    roots = []
    for collection in collections:
        roots.extend(collection.get('roots', []))
    return roots


def input_environments(collections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    environments = []
    for collection in collections:
        environment = collection.get('environment')
        if environment is not None:
            environments.append(environment)
    return environments


def render_metric(stat: dict[str, Any] | None) -> str:
    if stat is None:
        return 'NA'
    return (
        f"{fmt(stat.get('last'), 3)} "
        f"[min {fmt(stat.get('min'), 3)}, max {fmt(stat.get('max'), 3)}]"
    )


def render_report(data: dict[str, Any], title: str) -> str:
    roots = data['roots']
    records = [record for root in roots for record in root['records']]
    lines = [
        f'# {title}',
        '',
        f'- Generated at: `{data["generated_at"]}`',
        f'- Roots scanned: {len(roots)}',
        f'- Runs parsed: {len(records)}',
        '',
        '## Environment',
        '',
    ]
    env = data.get('environment')
    if env:
        lines.extend([
            f'- Python: `{env["python"].splitlines()[0]}`',
            f'- Repo: `{env["repo_path"]}`',
            f'- Git head: `{env["git_head"]}`',
            f'- Git status: `{(env["git_status"] or "").replace(chr(10), "; ")}`',
            f'- Packages: `{json.dumps(env["packages"], sort_keys=True)}`',
            '',
            '### Dataset Hashes',
            '',
            '| File | Bytes | SHA256 |',
            '| --- | ---: | --- |',
        ])
        for name, info in env['dataset_hashes']['files'].items():
            lines.append(f"| {name} | {info.get('bytes', 'missing')} | `{info.get('sha256', 'missing')}` |")
        lines.append('')

    source_envs = data.get('source_environments') or []
    if source_envs:
        lines.extend(['### Source Environments From Imported Records', ''])
        for source_env in source_envs:
            lines.extend([
                f"- Repo: `{source_env['repo_path']}`",
                f"  - Python: `{source_env['python'].splitlines()[0]}`",
                f"  - Git head: `{source_env['git_head']}`",
                f"  - Packages: `{json.dumps(source_env['packages'], sort_keys=True)}`",
            ])
        lines.append('')

    lines.extend([
        '## Paper Metric',
        '',
        '| Root | Task | Seeds | Paper success % mean +/- sd | Paper target % | Delta vs target | Final success % mean +/- sd |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ])
    for label in sorted({record['root_label'] for record in records}):
        for task in sorted(TARGET_SUCCESS_PCT):
            group = [r for r in records if r['root_label'] == label and r['task'] == task]
            if not group:
                continue
            complete = [r for r in group if r['paper_success'] is not None]
            paper_mean, paper_sd = mean_sd([
                r['paper_success'] * 100 if r['paper_success'] is not None else None
                for r in complete
            ])
            final_mean, final_sd = mean_sd([
                r['final_success'] * 100 if r['final_success'] is not None else None
                for r in group
            ])
            target_mean, target_sd = TARGET_SUCCESS_PCT[task]
            delta = None if paper_mean is None else paper_mean - target_mean
            lines.append(
                f'| {label} | {task} | {len(complete)} | {fmt(paper_mean)} +/- {fmt(paper_sd)} '
                f'| {target_mean:.1f} +/- {target_sd:.1f} | {fmt(delta)} '
                f'| {fmt(final_mean)} +/- {fmt(final_sd)} |'
            )

    lines.extend([
        '',
        '## Run Health',
        '',
        '| Root | Task | Seed | Best % | Paper % | Final % | Grad norm | Val critic loss | Actor MSE | Health flags | Run dir |',
        '| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |',
    ])
    for record in sorted(records, key=lambda r: (r['root_label'], r['task'] or 0, r['seed'] or 0)):
        metrics = record['metrics']
        lines.append(
            f"| {record['root_label']} | {record['task']} | {record['seed']} "
            f"| {fmt(record['best_success'] * 100 if record['best_success'] is not None else None)} "
            f"| {fmt(record['paper_success'] * 100 if record['paper_success'] is not None else None)} "
            f"| {fmt(record['final_success'] * 100 if record['final_success'] is not None else None)} "
            f"| {render_metric(metrics.get('training/grad/norm'))} "
            f"| {render_metric(metrics.get('validation/critic/critic_loss'))} "
            f"| {render_metric(metrics.get('training/actor/mse'))} "
            f"| {'; '.join(record['health_flags']) or 'none'} "
            f"| `{record['run_dir']}` |"
        )

    lines.extend([
        '',
        '## Protocol Checks',
        '',
        '| Root | Task | Seed | Protocol issues | Command-shape extras |',
        '| --- | ---: | ---: | --- | --- |',
    ])
    for record in sorted(records, key=lambda r: (r['root_label'], r['task'] or 0, r['seed'] or 0)):
        extras = []
        for key in ['enable_tensorboard', 'checkpoint_interval', 'checkpoint_at_end', 'resume_from_checkpoint']:
            if record.get(key) not in (None, 0, 0.0, False):
                extras.append(f'{key}={record.get(key)}')
        lines.append(
            f"| {record['root_label']} | {record['task']} | {record['seed']} "
            f"| {'; '.join(record['protocol_issues']) or 'none'} "
            f"| {'; '.join(extras) or 'none'} |"
        )

    lines.extend(['', '## Eval Curves', ''])
    for record in sorted(records, key=lambda r: (r['root_label'], r['task'] or 0, r['seed'] or 0)):
        curve = ', '.join(f"{step}:{fmt(value * 100 if value is not None else None, 0)}" for step, value in record['success_curve'])
        lines.append(f"- `{record['root_label']}` task {record['task']} seed {record['seed']}: {curve}")

    lines.extend(['', '## Slurm And Errors', ''])
    for root in roots:
        logs = root.get('slurm', {}).get('logs', [])
        errors = root.get('slurm', {}).get('error_hits', [])
        lines.append(f"- `{root['label']}`: {len(logs)} Slurm log summaries, {len(errors)} error-pattern hits.")
        for entry in logs[:12]:
            gpu = entry.get('gpu') or {}
            lines.append(
                f"  - job `{entry.get('job')}` host `{entry.get('host')}` repo `{entry.get('repo')}` "
                f"driver `{gpu.get('driver')}` cuda `{gpu.get('cuda')}`"
            )
        for hit in errors[:10]:
            lines.append(f"  - error `{hit['path']}:{hit['line']}`: {hit['text']}")

    lines.extend(['', '## Notes', ''])
    lines.append('- Paper metric here is mean success at 1.4M, 1.45M, and 1.5M total steps.')
    lines.append('- Best success is diagnostic only and is not the paper metric.')
    lines.append('- Validation spikes are flagged as diagnostics; they are not by themselves proof of failed policy learning.')
    lines.append('')
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', action='append', default=[], help='Run root as LABEL=PATH or PATH.')
    parser.add_argument('--records-input', action='append', type=Path, default=[])
    parser.add_argument('--records-output', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--title', default='OGBench Cube-Single InFOM Paper-Reproduction Diagnosis')
    parser.add_argument('--repo-path', type=Path, default=Path.cwd())
    parser.add_argument('--dataset-dir', type=Path, default=Path('~/.ogbench/data'))
    args = parser.parse_args()

    collections = [load_records(path.expanduser()) for path in args.records_input]
    roots = combine_collections(collections)
    for value in args.run_root:
        label, path = parse_labeled_path(value)
        roots.append(collect_root(label, path))

    data = {
        'generated_at': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        'environment': collect_environment(args.repo_path.expanduser().resolve(), args.dataset_dir.expanduser()),
        'source_environments': input_environments(collections),
        'roots': roots,
    }

    if args.records_output:
        args.records_output.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.records_output.expanduser().write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')

    report = render_report(data, args.title)
    if args.output:
        args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.output.expanduser().write_text(report)
    else:
        print(report)


if __name__ == '__main__':
    main()
