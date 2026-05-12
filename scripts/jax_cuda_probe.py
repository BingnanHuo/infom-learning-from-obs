#!/usr/bin/env python3
"""Probe whether JAX can initialize and execute on the allocated CUDA device."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd: list[str]) -> dict:
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return {
        'cmd': cmd,
        'returncode': proc.returncode,
        'output': proc.stdout,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional JSON output path.')
    parser.add_argument('--matrix-size', type=int, default=2048)
    args = parser.parse_args()

    result = {
        'ok': False,
        'host': socket.gethostname(),
        'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'python': sys.version,
        'platform': platform.platform(),
        'env': {
            key: os.environ.get(key)
            for key in [
                'CUDA_VISIBLE_DEVICES',
                'SLURM_JOB_ID',
                'SLURM_JOB_GPUS',
                'SLURM_JOB_NODELIST',
                'SLURM_JOB_PARTITION',
                'SLURM_STEP_GPUS',
                'XLA_PYTHON_CLIENT_PREALLOCATE',
            ]
        },
        'commands': {
            'nvidia_smi_l': run_command(['nvidia-smi', '-L']),
            'nvidia_smi_query': run_command([
                'nvidia-smi',
                '--query-gpu=index,name,pci.bus_id,driver_version,memory.total,memory.used,utilization.gpu',
                '--format=csv,noheader',
            ]),
        },
    }

    try:
        import jax
        import jax.numpy as jnp
        import jaxlib

        result['jax'] = {
            'jax': getattr(jax, '__version__', 'unknown'),
            'jaxlib': getattr(jaxlib, '__version__', 'unknown'),
            'default_backend': jax.default_backend(),
            'devices': [str(device) for device in jax.devices()],
        }
        size = args.matrix_size
        x = jnp.ones((size, size), dtype=jnp.float32)
        start = time.time()
        y = (x @ x).block_until_ready()
        elapsed = time.time() - start
        result['matmul'] = {
            'size': size,
            'seconds': elapsed,
            'checksum': float(jnp.mean(y)),
        }
        result['ok'] = True
    except Exception as exc:
        result['error'] = repr(exc)

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n')

    return 0 if result['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
