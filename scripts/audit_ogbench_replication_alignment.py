#!/usr/bin/env python3
"""Audit OGBench cube-single InFOM replication alignment.

This script checks the reproducibility-critical surfaces separately from run
summaries: upstream code identity, paper protocol constants, current-branch
deltas, dataset availability/provenance, and existing diagnosis evidence.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


PAPER_URL = 'https://arxiv.org/abs/2506.08902'
UPSTREAM_URL = 'https://github.com/chongyi-zheng/infom'
OGBENCH_DATA_URL = 'https://rail.eecs.berkeley.edu/datasets/ogbench'
UPSTREAM_REF = 'upstream/main'
PAPER_PROTOCOL = {
    'pretraining_steps': 1_000_000,
    'finetuning_steps': 500_000,
    'pretraining_size': 1_000_000,
    'finetuning_size': 500_000,
    'eval_interval': 50_000,
    'eval_episodes': 50,
    'paper_metric_finetuning_steps': [400_000, 450_000, 500_000],
    'lr': 3e-4,
    'batch_size': 256,
    'latent_dim': 512,
    'expectile': 0.95,
    'kl_weight': 0.05,
    'alpha': 30,
    'discount': 0.99,
    'tau': 0.005,
    'num_flow_goals': 16,
    'num_flow_steps': 10,
    'actor_freq': 4,
    'q_agg': 'min',
    'normalize_q_loss': False,
}
PAPER_TARGETS = {
    1: (92.5, 4.0),
    2: (78.4, 12.3),
    3: (56.4, 36.9),
    4: (91.5, 14.2),
    5: (70.0, 39.1),
}
DATASET_FILES = [
    'cube-single-play-v0.npz',
    'cube-single-play-v0-val.npz',
    'cube-single-play-ft-v0.npz',
    'cube-single-play-ft-v0-val.npz',
]
PACKAGE_NAMES = [
    ('jax', 'jax'),
    ('jaxlib', 'jaxlib'),
    ('flax', 'flax'),
    ('optax', 'optax'),
    ('ogbench', 'ogbench'),
    ('gymnasium', 'gymnasium'),
    ('numpy', 'numpy'),
    ('distrax', 'distrax'),
    ('ml_collections', 'ml_collections'),
    ('mujoco', 'mujoco'),
    ('dm_control', 'dm-control'),
]
REPLICATION_RELEVANT_FILES = [
    'main.py',
    'agents/infom.py',
    'envs/env_utils.py',
    'envs/ogbench_utils.py',
    'utils/datasets.py',
    'utils/evaluation.py',
    'data_gen_scripts/generate_ogbench_manispace.py',
    'requirements.txt',
    'README.md',
]


def run(cmd: list[str], cwd: Path, check: bool = False) -> str:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=check,
    )
    return proc.stdout.strip()


def git_show(repo: Path, ref: str, path: str) -> str:
    return run(['git', 'show', f'{ref}:{path}'], repo)


def parse_define_integer(text: str, name: str) -> int | None:
    match = re.search(rf"flags\.DEFINE_integer\('{re.escape(name)}',\s*([0-9_]+)", text)
    return int(match.group(1).replace('_', '')) if match else None


def parse_config_value(text: str, name: str) -> Any:
    patterns = [
        rf"{re.escape(name)}=([0-9.eE+-]+)",
        rf"{re.escape(name)}='([^']+)'",
        rf"{re.escape(name)}=(True|False)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = match.group(1)
            if value in {'True', 'False'}:
                return value == 'True'
            try:
                return float(value) if any(marker in value for marker in '.eE') else int(value)
            except ValueError:
                return value
    return None


def collect_upstream_protocol(repo: Path) -> dict[str, Any]:
    main_py = git_show(repo, UPSTREAM_REF, 'main.py')
    infom_py = git_show(repo, UPSTREAM_REF, 'agents/infom.py')
    return {
        'pretraining_steps': parse_define_integer(main_py, 'pretraining_steps'),
        'pretraining_size': parse_define_integer(main_py, 'pretraining_size'),
        'finetuning_steps': parse_define_integer(main_py, 'finetuning_steps'),
        'finetuning_size': parse_define_integer(main_py, 'finetuning_size'),
        'eval_interval': parse_define_integer(main_py, 'eval_interval'),
        'eval_episodes': parse_define_integer(main_py, 'eval_episodes'),
        'save_interval': parse_define_integer(main_py, 'save_interval'),
        'obs_norm_type': 'normal' if "flags.DEFINE_string('obs_norm_type', 'normal'" in main_py else None,
        'lr': parse_config_value(infom_py, 'lr'),
        'batch_size': parse_config_value(infom_py, 'batch_size'),
        'latent_dim': parse_config_value(infom_py, 'latent_dim'),
        'discount': parse_config_value(infom_py, 'discount'),
        'tau': parse_config_value(infom_py, 'tau'),
        'num_flow_goals': parse_config_value(infom_py, 'num_flow_goals'),
        'num_flow_steps': parse_config_value(infom_py, 'num_flow_steps'),
        'actor_freq': parse_config_value(infom_py, 'actor_freq'),
        'q_agg': parse_config_value(infom_py, 'q_agg'),
        'normalize_q_loss': parse_config_value(infom_py, 'normalize_q_loss'),
    }


def head_url(url: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, method='HEAD')
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return {
                'url': url,
                'status': response.status,
                'content_length': response.headers.get('Content-Length'),
                'error': None,
            }
    except urllib.error.HTTPError as exc:
        return {'url': url, 'status': exc.code, 'content_length': None, 'error': str(exc)}
    except OSError as exc:
        return {'url': url, 'status': None, 'content_length': None, 'error': str(exc)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def local_dataset_manifest(dataset_dir: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for name in DATASET_FILES:
        path = dataset_dir / name
        if not path.exists():
            files[name] = {'exists': False}
            continue
        info: dict[str, Any] = {
            'exists': True,
            'bytes': path.stat().st_size,
            'sha256': sha256_file(path),
            'arrays': {},
        }
        try:
            import numpy as np

            with np.load(path) as data:
                for key in data.files:
                    array = data[key]
                    info['arrays'][key] = {'shape': list(array.shape), 'dtype': str(array.dtype)}
        except Exception as exc:  # pragma: no cover - diagnostic only.
            info['array_error'] = str(exc)
        files[name] = info
    return {'dataset_dir': str(dataset_dir), 'files': files}


def detect_generator_seed_controls(text: str) -> dict[str, bool]:
    return {
        'declares_seed_flag': "DEFINE_integer('seed'" in text,
        'seeds_numpy': 'np.random.seed' in text,
        'seeds_env_reset': 'env.reset(seed=' in text,
        'seeds_action_space': 'action_space.seed' in text,
    }


def collect_current_deltas(repo: Path) -> dict[str, Any]:
    diff_name_status = run(['git', 'diff', '--name-status', UPSTREAM_REF, '--', *REPLICATION_RELEVANT_FILES], repo)
    diff_stat = run(['git', 'diff', '--stat', UPSTREAM_REF, '--', *REPLICATION_RELEVANT_FILES], repo)
    current_status = run(['git', 'status', '--short', '--branch'], repo)
    current_head = run(['git', 'rev-parse', 'HEAD'], repo)
    upstream_head = run(['git', 'rev-parse', UPSTREAM_REF], repo)
    upstream_remote = run(['git', 'ls-remote', 'upstream', 'HEAD', 'refs/heads/main'], repo)
    main_text = (repo / 'main.py').read_text()
    infom_text = (repo / 'agents' / 'infom.py').read_text()
    generator_text = (repo / 'data_gen_scripts' / 'generate_ogbench_manispace.py').read_text()
    upstream_generator_text = git_show(repo, UPSTREAM_REF, 'data_gen_scripts/generate_ogbench_manispace.py')
    return {
        'current_head': current_head,
        'upstream_head': upstream_head,
        'upstream_remote': upstream_remote,
        'git_status': current_status,
        'diff_name_status': diff_name_status.splitlines() if diff_name_status else [],
        'diff_stat': diff_stat,
        'current_feature_flags': {
            'tensorboard': 'enable_tensorboard' in main_text,
            'checkpoint_resume': 'resume_from_checkpoint' in main_text,
            'bridge_loss': 'bridge_loss_weight' in infom_text,
            'dataset_dir': 'dataset_dir' in main_text,
        },
        'generator_seed_controls': detect_generator_seed_controls(generator_text),
        'upstream_generator_seed_controls': detect_generator_seed_controls(upstream_generator_text),
    }


def collect_package_versions() -> dict[str, str]:
    packages: dict[str, str] = {}
    for label, package_name in PACKAGE_NAMES:
        try:
            packages[label] = version(package_name)
        except PackageNotFoundError:
            packages[label] = 'not installed'
    return packages


def load_existing_result_records(report_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(report_dir.glob('diagnosis*.records.json')):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for root in data.get('roots', []):
            for record in root.get('records', []):
                records.append(
                    {
                        'source': str(path),
                        'root_label': record.get('root_label') or root.get('label'),
                        'task': record.get('task'),
                        'seed': record.get('seed'),
                        'paper_success_pct': (
                            record['paper_success'] * 100
                            if record.get('paper_success') is not None
                            else None
                        ),
                        'final_success_pct': (
                            record['final_success'] * 100
                            if record.get('final_success') is not None
                            else None
                        ),
                        'best_success_pct': (
                            record['best_success'] * 100
                            if record.get('best_success') is not None
                            else None
                        ),
                        'protocol_issues': record.get('protocol_issues') or [],
                        'health_flags': record.get('health_flags') or [],
                        'run_dir': record.get('run_dir'),
                    }
                )
    deduped: dict[str, dict[str, Any]] = {}
    for record in records:
        key = record.get('run_dir') or f"{record['source']}:{record.get('task')}:{record.get('seed')}"
        deduped[key] = record
    return list(deduped.values())


def status_cell(ok: bool | None) -> str:
    if ok is True:
        return 'aligned'
    if ok is False:
        return 'gap'
    return 'unknown'


def fmt(value: Any) -> str:
    if value is None:
        return 'NA'
    if isinstance(value, float):
        return f'{value:g}'
    return str(value)


def alignment_rows(upstream_protocol: dict[str, Any], deltas: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for key in [
        'pretraining_steps',
        'finetuning_steps',
        'pretraining_size',
        'finetuning_size',
        'eval_interval',
        'eval_episodes',
        'lr',
        'batch_size',
        'latent_dim',
        'discount',
        'tau',
        'num_flow_goals',
        'num_flow_steps',
        'actor_freq',
        'q_agg',
        'normalize_q_loss',
    ]:
        expected = PAPER_PROTOCOL.get(key)
        observed = upstream_protocol.get(key)
        rows.append(
            {
                'dimension': key,
                'paper_expected': fmt(expected),
                'upstream_observed': fmt(observed),
                'status': status_cell(observed == expected),
                'action': 'Use clean upstream command path.' if observed == expected else 'Inspect paper/code mismatch.',
            }
        )

    rows.extend(
        [
            {
                'dimension': 'cube_single_overrides',
                'paper_expected': 'expectile=0.95, kl_weight=0.05, alpha=30',
                'upstream_observed': 'set by run command, not default config',
                'status': 'aligned when command includes overrides',
                'action': 'Reject runs missing these flags.',
            },
            {
                'dimension': 'current_branch_extras',
                'paper_expected': 'no bridge/checkpoint/TensorBoard extensions',
                'upstream_observed': ', '.join(k for k, v in deltas['current_feature_flags'].items() if v),
                'status': 'gap for exact replication',
                'action': 'Use clean upstream for replication claims.',
            },
            {
                'dimension': 'generator_seed',
                'paper_expected': 'reproducible fine-tune data or published hash',
                'upstream_observed': json.dumps(deltas['upstream_generator_seed_controls'], sort_keys=True),
                'status': 'gap',
                'action': 'Use patched generator for sensitivity tests; do not call generated data paper-identical.',
            },
        ]
    )
    return rows


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        '# OGBench Cube-Single InFOM Replication Alignment Audit',
        '',
        f"- Generated: `{report['generated_at']}`",
        f"- Paper: {PAPER_URL}",
        f"- Upstream repo: {UPSTREAM_URL}",
        f"- OGBench dataset host: {OGBENCH_DATA_URL}",
        f"- Current HEAD: `{report['deltas']['current_head']}`",
        f"- Upstream `{UPSTREAM_REF}`: `{report['deltas']['upstream_head']}`",
        '',
        '## Local Environment',
        '',
        f"- Python: `{report['python'].splitlines()[0]}`",
        '',
        '| Package | Version |',
        '| --- | --- |',
    ]
    for name, package_version in report['packages'].items():
        lines.append(f'| {name} | {package_version} |')

    lines.extend(
        [
            '',
        '## Alignment Matrix',
        '',
        '| Dimension | Paper expected | Upstream/current observed | Status | Action |',
        '| --- | --- | --- | --- | --- |',
        ]
    )
    for row in report['alignment_rows']:
        lines.append(
            '| {dimension} | {paper_expected} | {upstream_observed} | {status} | {action} |'.format(
                **{key: str(value).replace('\n', '<br>') for key, value in row.items()}
            )
        )

    lines.extend(
        [
            '',
            '## Dataset Availability',
            '',
            '| File | Remote status | Remote bytes | Local bytes | Local SHA256 | Notes |',
            '| --- | ---: | ---: | ---: | --- | --- |',
        ]
    )
    remote_by_name = {Path(item['url']).name: item for item in report['remote_datasets']}
    for name, local in report['local_datasets']['files'].items():
        remote = remote_by_name.get(name, {})
        notes = []
        if remote.get('status') == 404:
            notes.append('not downloadable from official host')
        if not local.get('exists'):
            notes.append('missing locally')
        elif '-ft-' in name:
            notes.append('locally generated unless external artifact supplied')
        lines.append(
            f"| {name} | {fmt(remote.get('status'))} | {fmt(remote.get('content_length'))} "
            f"| {fmt(local.get('bytes'))} | `{fmt(local.get('sha256'))}` | {'; '.join(notes) or 'official download path'} |"
        )

    lines.extend(
        [
            '',
            '## Current Branch Deltas',
            '',
            '```text',
            report['deltas']['diff_stat'] or 'No replication-relevant diff from upstream.',
            '```',
            '',
            '## Existing Result Evidence',
            '',
            '| Source | Root | Task | Seed | Paper success % | Final success % | Best success % | Notes |',
            '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |',
        ]
    )
    for record in report['result_records'][-20:]:
        notes = []
        if record['protocol_issues']:
            notes.append('protocol issues: ' + '; '.join(record['protocol_issues']))
        if record['health_flags']:
            notes.append('health flags: ' + '; '.join(record['health_flags'][:2]))
        lines.append(
            f"| {Path(record['source']).name} | {fmt(record['root_label'])} | {fmt(record['task'])} "
            f"| {fmt(record['seed'])} | {fmt(record['paper_success_pct'])} "
            f"| {fmt(record['final_success_pct'])} | {fmt(record['best_success_pct'])} "
            f"| {'; '.join(notes) or 'none'} |"
        )
    if not report['result_records']:
        lines.append('| none | NA | NA | NA | NA | NA | NA | no local diagnosis records found |')

    lines.extend(
        [
            '',
            '## Conclusions',
            '',
            '- Hyperparameters and evaluation protocol are aligned when using clean upstream plus the cube-single command overrides.',
            '- The fine-tuning dataset is not proven paper-identical: official `*-ft-*` files are unavailable from the OGBench host, and upstream generation is not seeded.',
            '- Current-branch additions are useful for operations, but exact replication evidence should come from clean upstream runs.',
            '- Existing task1 evidence shows mid-run success can reach paper-level performance before collapsing, so late-stage stability and fine-tune data sensitivity are the immediate diagnosis targets.',
            '',
            '## Recommended Next Runs',
            '',
            '1. Keep pending clean-upstream task4/task5 runs as the sanity anchor.',
            '2. Generate 2-3 seeded cube-single ft datasets with the patched generator and compare task1 seed0 curves under clean upstream training code.',
            '3. If seeded dataset sensitivity explains task1, document provenance and choose stable tasks/datasets for contribution comparisons.',
            '4. If it does not, run a package-version A/B after pinning a paper-candidate environment.',
            '',
        ]
    )
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path.cwd())
    parser.add_argument('--dataset-dir', type=Path, default=Path('~/.ogbench/data'))
    parser.add_argument('--report-dir', type=Path, default=Path('runs/reports/ogbench_paper_repro'))
    parser.add_argument('--output-md', type=Path)
    parser.add_argument('--output-json', type=Path)
    parser.add_argument('--url-timeout', type=float, default=10.0)
    parser.add_argument('--skip-url-check', action='store_true')
    args = parser.parse_args()

    repo = args.repo_root.expanduser().resolve()
    report_dir = (repo / args.report_dir).resolve() if not args.report_dir.is_absolute() else args.report_dir
    dataset_dir = args.dataset_dir.expanduser().resolve()
    timestamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S')
    output_md = args.output_md or report_dir / f'alignment_audit_{timestamp}.md'
    output_json = args.output_json or report_dir / f'alignment_audit_{timestamp}.json'

    upstream_protocol = collect_upstream_protocol(repo)
    deltas = collect_current_deltas(repo)
    remote_datasets = []
    if not args.skip_url_check:
        remote_datasets = [head_url(f'{OGBENCH_DATA_URL}/{name}', args.url_timeout) for name in DATASET_FILES]
    else:
        remote_datasets = [{'url': f'{OGBENCH_DATA_URL}/{name}', 'status': None, 'content_length': None, 'error': 'skipped'} for name in DATASET_FILES]

    report = {
        'generated_at': dt.datetime.now(dt.timezone.utc).isoformat(),
        'paper_url': PAPER_URL,
        'upstream_url': UPSTREAM_URL,
        'paper_protocol': PAPER_PROTOCOL,
        'paper_targets': PAPER_TARGETS,
        'upstream_protocol': upstream_protocol,
        'deltas': deltas,
        'alignment_rows': alignment_rows(upstream_protocol, deltas),
        'remote_datasets': remote_datasets,
        'local_datasets': local_dataset_manifest(dataset_dir),
        'result_records': load_existing_result_records(report_dir),
        'python': sys.version,
        'packages': collect_package_versions(),
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    output_md.write_text(render_markdown(report))
    print(output_md)
    print(output_json)


if __name__ == '__main__':
    main()
