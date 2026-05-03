#!/usr/bin/env python3
"""Summarize matched OGBench bridge runs from CSV artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


EVAL_RETURN = 'evaluation/episode.return'
EVAL_SUCCESS = 'evaluation/episode.success'
BRIDGE_TRAIN_LOSS = 'training/bridge/weighted_bridge_loss'
BRIDGE_VAL_LOSS = 'validation/bridge/weighted_bridge_loss'
ERROR_PATTERN = re.compile(
    r'Traceback|RuntimeError|OutOfMemory|out of memory|OOM|CANCELLED|TIMEOUT|DUE TO TIME|FAILED|Error'
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root', type=Path, help='Root directory containing run subdirectories.')
    parser.add_argument('--output', type=Path, help='Optional Markdown output path.')
    parser.add_argument('--title', default='OGBench Bridge Run Summary', help='Markdown report title.')
    parser.add_argument('--slurm-root', type=Path, help='Optional Slurm log directory. Defaults to run_root/slurm.')
    return parser.parse_args()


def read_csv_rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def read_json(path):
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def to_float(value):
    if value in (None, ''):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def fmt(value, digits=3):
    if value is None:
        return 'NA'
    return f'{value:.{digits}f}'


def mean_sd(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def infer_variant(run_dir, flags):
    text = str(run_dir).lower()
    if 'bridge_aware' in text or 'bridge-aware' in text:
        return 'bridge-aware'
    if 'ego_only' in text or 'ego-only' in text:
        return 'ego-only'
    agent = flags.get('agent') or {}
    if float(agent.get('bridge_loss_weight') or 0.0) > 0.0:
        return 'bridge-aware'
    return 'ego-only'


def checkpoint_count(run_dir):
    return len(list(run_dir.glob('params_*.pkl')))


def best_eval_record(run_dir):
    path = run_dir / 'best_eval.json'
    data = read_json(path)
    if not data:
        return None
    return data


def parse_run(run_dir):
    eval_path = run_dir / 'finetuning_eval.csv'
    if not eval_path.exists():
        return None

    flags = read_json(run_dir / 'flags.json')
    eval_rows = read_csv_rows(eval_path)
    if not eval_rows:
        return None

    returns = [to_float(row.get(EVAL_RETURN)) for row in eval_rows]
    successes = [to_float(row.get(EVAL_SUCCESS)) for row in eval_rows]
    steps = [int(float(row['step'])) for row in eval_rows if row.get('step')]
    returns_clean = [v for v in returns if v is not None]
    successes_clean = [v for v in successes if v is not None]
    agent = flags.get('agent') or {}

    run_name = run_dir.name
    seed = flags.get('seed')
    if seed is None:
        match = re.search(r'sd(\d+)', run_name)
        seed = int(match.group(1)) if match else None

    job_id = None
    match = re.search(r'_s_(\d+)', run_name)
    if match:
        job_id = match.group(1)

    record = {
        'run_dir': str(run_dir),
        'run_name': run_name,
        'variant': infer_variant(run_dir, flags),
        'seed': int(seed) if seed is not None else None,
        'job_id': job_id,
        'eval_rows': len(eval_rows),
        'first_eval_step': min(steps) if steps else None,
        'final_step': max(steps) if steps else None,
        'final_return': returns[-1],
        'final_success': successes[-1],
        'best_return': max(returns_clean) if returns_clean else None,
        'best_success': max(successes_clean) if successes_clean else None,
        'bridge_loss_weight': float(agent.get('bridge_loss_weight') or 0.0),
        'checkpoint_count': checkpoint_count(run_dir),
        'best_eval': best_eval_record(run_dir),
    }

    pretrain_path = run_dir / 'pretraining_train.csv'
    if pretrain_path.exists():
        rows = read_csv_rows(pretrain_path)
        if rows and BRIDGE_TRAIN_LOSS in rows[0]:
            train_losses = [to_float(row.get(BRIDGE_TRAIN_LOSS)) for row in rows]
            val_losses = [to_float(row.get(BRIDGE_VAL_LOSS)) for row in rows]
            train_losses = [v for v in train_losses if v is not None]
            val_losses = [v for v in val_losses if v is not None]
            record['bridge_train_loss_first'] = train_losses[0] if train_losses else None
            record['bridge_train_loss_last'] = train_losses[-1] if train_losses else None
            record['bridge_val_loss_last'] = val_losses[-1] if val_losses else None

    return record


def find_run_dirs(root):
    return sorted(path.parent for path in root.rglob('finetuning_eval.csv'))


def collect_slurm_hits(slurm_root):
    hits = []
    if slurm_root is None or not slurm_root.exists():
        return hits
    for path in sorted(slurm_root.glob('*')):
        if not path.is_file():
            continue
        try:
            for line_no, line in enumerate(path.read_text(errors='replace').splitlines(), start=1):
                if ERROR_PATTERN.search(line):
                    hits.append((str(path), line_no, line[:240]))
        except OSError as exc:
            hits.append((str(path), 0, f'Could not read log: {exc}'))
    return hits


def aggregate(records):
    lines = [
        '| Variant | Seeds | Final return mean +/- sd | Final success mean +/- sd | Best return mean +/- sd | Best success mean +/- sd |',
        '| --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for variant in sorted({r['variant'] for r in records}):
        group = [r for r in records if r['variant'] == variant]
        final_return = mean_sd([r['final_return'] for r in group])
        final_success = mean_sd([r['final_success'] for r in group])
        best_return = mean_sd([r['best_return'] for r in group])
        best_success = mean_sd([r['best_success'] for r in group])
        lines.append(
            f'| {variant} | {len(group)} | {fmt(final_return[0], 2)} +/- {fmt(final_return[1], 2)} '
            f'| {fmt(final_success[0], 3)} +/- {fmt(final_success[1], 3)} '
            f'| {fmt(best_return[0], 2)} +/- {fmt(best_return[1], 2)} '
            f'| {fmt(best_success[0], 3)} +/- {fmt(best_success[1], 3)} |'
        )
    return '\n'.join(lines)


def matched_deltas(records):
    by_seed = {}
    for record in records:
        by_seed.setdefault(record['seed'], {})[record['variant']] = record

    lines = [
        '| Seed | Final return delta | Final success delta | Best return delta | Best success delta |',
        '| ---: | ---: | ---: | ---: | ---: |',
    ]
    deltas = []
    for seed in sorted(by_seed):
        ego = by_seed[seed].get('ego-only')
        bridge = by_seed[seed].get('bridge-aware')
        if ego is None or bridge is None:
            continue
        row = {
            'seed': seed,
            'final_return': bridge['final_return'] - ego['final_return'],
            'final_success': bridge['final_success'] - ego['final_success'],
            'best_return': bridge['best_return'] - ego['best_return'],
            'best_success': bridge['best_success'] - ego['best_success'],
        }
        deltas.append(row)
        lines.append(
            f"| {seed} | {fmt(row['final_return'], 2)} | {fmt(row['final_success'], 3)} "
            f"| {fmt(row['best_return'], 2)} | {fmt(row['best_success'], 3)} |"
        )

    if not deltas:
        return 'No matched ego-only and bridge-aware seed pairs found.'

    final_return = mean_sd([d['final_return'] for d in deltas])
    final_success = mean_sd([d['final_success'] for d in deltas])
    best_return = mean_sd([d['best_return'] for d in deltas])
    best_success = mean_sd([d['best_success'] for d in deltas])
    lines.append('')
    lines.append(
        f'Mean bridge-minus-ego deltas: final return {fmt(final_return[0], 2)} +/- {fmt(final_return[1], 2)}, '
        f'final success {fmt(final_success[0], 3)} +/- {fmt(final_success[1], 3)}, '
        f'best return {fmt(best_return[0], 2)} +/- {fmt(best_return[1], 2)}, '
        f'best success {fmt(best_success[0], 3)} +/- {fmt(best_success[1], 3)}.'
    )
    return '\n'.join(lines)


def per_seed_table(records):
    lines = [
        '| Variant | Seed | Job ID | Eval rows | Final step | Final return | Final success | Best return | Best success | Checkpoints | Best eval step |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for record in sorted(records, key=lambda r: (r['variant'], r['seed'])):
        best_eval = record['best_eval'] or {}
        lines.append(
            f"| {record['variant']} | {record['seed']} | {record['job_id'] or 'NA'} "
            f"| {record['eval_rows']} | {record['final_step']} | {fmt(record['final_return'], 2)} "
            f"| {fmt(record['final_success'], 3)} | {fmt(record['best_return'], 2)} "
            f"| {fmt(record['best_success'], 3)} | {record['checkpoint_count']} "
            f"| {best_eval.get('step', 'NA')} |"
        )
    return '\n'.join(lines)


def bridge_loss_table(records):
    bridge_records = [
        r for r in records
        if r['variant'] == 'bridge-aware' and 'bridge_train_loss_first' in r
    ]
    if not bridge_records:
        return 'No bridge loss columns found.'

    lines = [
        '| Seed | Train weighted first | Train weighted last | Val weighted last |',
        '| ---: | ---: | ---: | ---: |',
    ]
    for record in sorted(bridge_records, key=lambda r: r['seed']):
        lines.append(
            f"| {record['seed']} | {fmt(record.get('bridge_train_loss_first'), 10)} "
            f"| {fmt(record.get('bridge_train_loss_last'), 10)} "
            f"| {fmt(record.get('bridge_val_loss_last'), 10)} |"
        )
    return '\n'.join(lines)


def render_markdown(title, run_root, records, slurm_hits):
    lines = [
        f'# {title}',
        '',
        f'- Run root: `{run_root}`',
        f'- Run directories parsed: {len(records)}',
        f'- Slurm error pattern hits: {len(slurm_hits)}',
        '',
        '## Aggregate',
        '',
        aggregate(records),
        '',
        '## Matched Seed Deltas',
        '',
        matched_deltas(records),
        '',
        '## Per-Seed Runs',
        '',
        per_seed_table(records),
        '',
        '## Bridge Objective',
        '',
        bridge_loss_table(records),
    ]
    if slurm_hits:
        lines.extend(['', '## Slurm Error Pattern Hits', ''])
        for path, line_no, text in slurm_hits[:50]:
            lines.append(f'- `{path}:{line_no}`: {text}')
    lines.append('')
    return '\n'.join(lines)


def main():
    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    slurm_root = args.slurm_root
    if slurm_root is None:
        default_slurm = run_root / 'slurm'
        slurm_root = default_slurm if default_slurm.exists() else None
    elif slurm_root is not None:
        slurm_root = slurm_root.expanduser().resolve()

    records = [parse_run(path) for path in find_run_dirs(run_root)]
    records = [record for record in records if record is not None]
    if not records:
        raise SystemExit(f'No finetuning_eval.csv files found under {run_root}')

    markdown = render_markdown(args.title, run_root, records, collect_slurm_hits(slurm_root))
    if args.output:
        output = args.output.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown)
    else:
        print(markdown)


if __name__ == '__main__':
    main()
