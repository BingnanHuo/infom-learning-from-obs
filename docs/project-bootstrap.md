# Project Bootstrap

Bootstrapped on April 21, 2026 in the worktree `<repo_root>`.

## Repository State

- GitHub fork: `BingnanHuo/infom-learning-from-obs`
- Local branch: `bootstrap/infom-obs-foundation`
- `origin`: `https://github.com/BingnanHuo/infom-learning-from-obs.git`
- `upstream`: `https://github.com/chongyi-zheng/infom.git`

The local class PDFs that existed before git initialization were preserved under `docs/`.

## Environment

The tracked environment is `infom-obs` and is defined by [`environment.yml`](../environment.yml).

Installed bootstrap stack:

- `python=3.10.16`
- `glew`
- `mesalib`
- upstream `requirements.txt`, including CUDA-enabled JAX

Required runtime variables:

```bash
export PYTHONPATH=$(pwd)
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

## Smoke Checks

The following checks were run successfully in `infom-obs`:

- import `jax`, `flax`, `distrax`, `ml_collections`, and `ogbench`
- import `agents.infom`, `utils.encoders`, and `envs.env_utils`
- verify `jax.devices()` sees `CudaDevice(id=0)`
- verify `get_config().agent_name == "infom"`

`python main.py --helpfull` printed the full CLI help. It exits with status code `1` under `absl`, which is expected and not treated as a bootstrap failure.

## Deferred Work

Bootstrap intentionally does not yet:

- download large benchmark datasets
- run long training jobs
- modify the upstream agent implementation
- introduce Isaac Lab dependencies

The next implementation milestone is described in `docs/design/phase1-self-third-person-bridge.md`.
