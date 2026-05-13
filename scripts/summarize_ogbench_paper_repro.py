#!/usr/bin/env python3
"""Summarize OGBench paper-protocol reproduction runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


EVAL_RETURN = 'evaluation/episode.return'
EVAL_SUCCESS = 'evaluation/episode.success'
ERROR_PATTERN = re.compile(
    r'Traceback|RuntimeError|OutOfMemory|out of memory|\bOOM\b|CANCELLED|TIMEOUT|DUE TO TIME|FAILED|\bERROR\b|(^|\s)Error(:|\s)'
)
TARGET_SUCCESS_PCT = {
    1: (92.5, 4.0),
    2: (78.4, 12.3),
    3: (56.4, 36.9),
    4: (91.5, 14.2),
    5: (70.0, 39.1),
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def to_float(value: str | None) -> float | None:
    if value in (None, ''):
        return None
    try:
        return float(value)
    except ValueError:
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


def infer_task(flags: dict, run_dir: Path) -> int | None:
    env_name = str(flags.get('env_name') or run_dir)
    match = re.search(r'cube-single-play-singletask-task(\d+)-v0', env_name)
    return int(match.group(1)) if match else None


def infer_seed(flags: dict, run_dir: Path) -> int | None:
    if flags.get('seed') is not None:
        return int(flags['seed'])
    match = re.search(r'sd(\d+)', run_dir.name)
    return int(match.group(1)) if match else None


def parse_run(run_dir: Path) -> dict | None:
    eval_path = run_dir / 'finetuning_eval.csv'
    if not eval_path.exists():
        return None
    flags = read_json(run_dir / 'flags.json')
    rows = read_csv_rows(eval_path)
    if not rows:
        return None

    pretraining_steps = int(flags.get('pretraining_steps', 1_000_000))
    paper_steps = [pretraining_steps + 400_000, pretraining_steps + 450_000, pretraining_steps + 500_000]
    by_step = {int(float(row['step'])): row for row in rows if row.get('step')}
    paper_successes = [to_float(by_step.get(step, {}).get(EVAL_SUCCESS)) for step in paper_steps]
    paper_returns = [to_float(by_step.get(step, {}).get(EVAL_RETURN)) for step in paper_steps]
    final = rows[-1]
    agent = flags.get('agent') or {}
    return {
        'run_dir': str(run_dir),
        'task': infer_task(flags, run_dir),
        'seed': infer_seed(flags, run_dir),
        'env_name': flags.get('env_name'),
        'pretraining_steps': pretraining_steps,
        'finetuning_steps': int(flags.get('finetuning_steps', 500_000)),
        'eval_interval': int(flags.get('eval_interval', 50_000)),
        'eval_episodes': int(flags.get('eval_episodes', 50)),
        'agent_expectile': agent.get('expectile'),
        'agent_kl_weight': agent.get('kl_weight'),
        'agent_alpha': agent.get('alpha'),
        'eval_rows': len(rows),
        'final_step': int(float(final['step'])),
        'final_success': to_float(final.get(EVAL_SUCCESS)),
        'final_return': to_float(final.get(EVAL_RETURN)),
        'paper_steps': paper_steps,
        'missing_paper_steps': [step for step, value in zip(paper_steps, paper_successes) if value is None],
        'paper_success': statistics.mean([v for v in paper_successes if v is not None]) if all(v is not None for v in paper_successes) else None,
        'paper_return': statistics.mean([v for v in paper_returns if v is not None]) if all(v is not None for v in paper_returns) else None,
    }


def find_run_dirs(root: Path) -> list[Path]:
    return sorted(path.parent for path in root.rglob('finetuning_eval.csv'))


def collect_slurm_hits(slurm_root: Path | None) -> list[tuple[str, int, str]]:
    hits: list[tuple[str, int, str]] = []
    if slurm_root is None or not slurm_root.exists():
        return hits
    for path in sorted(slurm_root.glob('*')):
        if not path.is_file():
            continue
        try:
            lines = path.read_text(errors='replace').splitlines()
        except OSError as exc:
            hits.append((str(path), 0, f'Could not read log: {exc}'))
            continue
        for line_no, line in enumerate(lines, start=1):
            if ERROR_PATTERN.search(line):
                hits.append((str(path), line_no, line[:240]))
    return hits


def protocol_issues(record: dict) -> list[str]:
    issues = []
    checks = {
        'pretraining_steps': 1_000_000,
        'finetuning_steps': 500_000,
        'eval_interval': 50_000,
        'eval_episodes': 50,
    }
    for key, expected in checks.items():
        if record.get(key) != expected:
            issues.append(f'{key}={record.get(key)} expected {expected}')
    agent_checks = {
        'agent_expectile': 0.95,
        'agent_kl_weight': 0.05,
        'agent_alpha': 30,
    }
    for key, expected in agent_checks.items():
        value = record.get(key)
        if value is None or abs(float(value) - expected) > 1e-12:
            issues.append(f'{key}={value} expected {expected}')
    return issues


def render(records: list[dict], run_root: Path, slurm_hits: list[tuple[str, int, str]], title: str) -> str:
    lines = [
        f'# {title}',
        '',
        f'- Run root: `{run_root}`',
        f'- Runs parsed: {len(records)}',
        f'- Slurm error pattern hits: {len(slurm_hits)}',
        '',
        '## Paper Metric',
        '',
        '| Task | Completed seeds | Paper success % mean +/- sd | Paper target % | Delta vs target | Final success % mean +/- sd |',
        '| ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for task in sorted(TARGET_SUCCESS_PCT):
        group = [r for r in records if r['task'] == task]
        complete = [r for r in group if r['paper_success'] is not None]
        paper_mean, paper_sd = mean_sd([r['paper_success'] * 100 if r['paper_success'] is not None else None for r in complete])
        final_mean, final_sd = mean_sd([r['final_success'] * 100 if r['final_success'] is not None else None for r in group])
        target_mean, target_sd = TARGET_SUCCESS_PCT[task]
        delta = None if paper_mean is None else paper_mean - target_mean
        lines.append(
            f'| {task} | {len(complete)} | {fmt(paper_mean)} +/- {fmt(paper_sd)} '
            f'| {target_mean:.1f} +/- {target_sd:.1f} | {fmt(delta)} '
            f'| {fmt(final_mean)} +/- {fmt(final_sd)} |'
        )

    lines.extend([
        '',
        '## Per Run',
        '',
        '| Task | Seed | Eval rows | Final step | Paper success % | Final success % | Missing paper steps | Protocol issues | Run dir |',
        '| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |',
    ])
    for record in sorted(records, key=lambda r: (r['task'] is None, r['task'] or 0, r['seed'] is None, r['seed'] or 0)):
        issues = '; '.join(protocol_issues(record))
        missing = ','.join(str(step) for step in record['missing_paper_steps'])
        lines.append(
            f"| {record['task'] if record['task'] is not None else 'NA'} "
            f"| {record['seed'] if record['seed'] is not None else 'NA'} "
            f"| {record['eval_rows']} | {record['final_step']} "
            f"| {fmt(record['paper_success'] * 100 if record['paper_success'] is not None else None)} "
            f"| {fmt(record['final_success'] * 100 if record['final_success'] is not None else None)} "
            f"| {missing or 'none'} | {issues or 'none'} | `{record['run_dir']}` |"
        )

    if slurm_hits:
        lines.extend(['', '## Slurm Error Pattern Hits', ''])
        for path, line_no, text in slurm_hits[:80]:
            lines.append(f'- `{path}:{line_no}`: {text}')
    lines.append('')
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--slurm-root', type=Path)
    parser.add_argument('--title', default='OGBench Cube-Single Paper Reproduction Summary')
    args = parser.parse_args()

    run_root = args.run_root.expanduser().resolve()
    slurm_root = args.slurm_root.expanduser().resolve() if args.slurm_root else run_root / 'slurm'
    records = [parse_run(path) for path in find_run_dirs(run_root)]
    records = [record for record in records if record is not None]
    if not records:
        raise SystemExit(f'No finetuning_eval.csv files found under {run_root}')

    markdown = render(records, run_root, collect_slurm_hits(slurm_root), args.title)
    if args.output:
        args.output.expanduser().write_text(markdown)
    else:
        print(markdown)


if __name__ == '__main__':
    main()
