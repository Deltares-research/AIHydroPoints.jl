# CLAUDE.md (pybridge)

This subfolder is a standalone experiment area for mlflow and ray, connected to the
main AIHydroPoints repo. It has its own `pixi.toml`/`pixi.lock`, independent of the
top-level project. Make changes only within `pybridge/` — do not edit files elsewhere
in the repo as part of this experiment.

See `plan.md` for the current plan and status.

## Running Python

Python here is managed via pixi, not a system/global interpreter — the pixi
environment is what has `mlflow`, `ray`, etc. installed. Never invoke a bare
`python`/`python3` for anything in this folder. Instead, use the wrapper:

```bash
pybridge/bin/python script.py [args...]
```

This works from anywhere in the repo and requires no PATH changes. If
`pybridge/settings.sh` has already been sourced in the current shell
(`source pybridge/settings.sh`), `pybridge/bin` is on `PATH` and a plain
`python` also resolves to the wrapper.

## Code style

Unlike the top-level repo's default (terse, comment-sparse) style, code in
`pybridge/` should be documented generously:

- Every function gets a docstring explaining what it does and how it's meant
  to be used (not just its parameters) — enough that someone unfamiliar with
  this experiment could use the function correctly from the docstring alone.
- Larger blocks within a function get a short comment describing what that
  block does, even where the code is fairly readable on its own.

This is a deliberate override for this folder — it's an experimentation area
being iterated on and revisited, not settled library code, so the extra
context is worth the verbosity.

## Installing dependencies

Add packages to `pybridge/pixi.toml` under `[dependencies]`, then run
`pixi install --manifest-path pybridge/pixi.toml` to update `pixi.lock`.

## Commands

**Run a script:**
```bash
pybridge/bin/python path/to/script.py
```

**Open a Python REPL:**
```bash
pixi run --manifest-path pybridge/pixi.toml python
```

**mlflow tracking server:**
```bash
pybridge/bin/mlflow-server-start [port]   # background, logs to pybridge/run/
pybridge/bin/mlflow-server-stop
```

**Train a model, logged to mlflow:**
```bash
pybridge/bin/python pybridge/mlflow_train.py path/to/settings.toml
```

**Parameter sweep, logged to mlflow** (mirrors `scripts/parameter_sweep.jl`,
same `sweeps/<experiment>/` output — see `plan.md`):
```bash
pybridge/bin/python pybridge/mlflow_sweep.py \
    [base.toml] [dotted.param.path] [v1,v2,v3] [nrepeats] [experiment] [--continue|--overwrite]
```

**Permanently purge mlflow experiments/runs deleted via the UI** (the UI's
delete is always a soft delete; this is the only way to actually free the
space/name):
```bash
pybridge/bin/mlflow-gc [--older-than DURATION]   # default: 1d
```
