# Script usage

Three entry points cover the day-to-day workflow: train one model, run
inference with a trained model, and sweep one setting across several values.
All three are TOML-driven — see [settings reference](settings.md) for the
config file format.

---

## Train a model

```bash
bin/train path/to/settings.toml [--continue|--overwrite]
```

(equivalently `pixi run julia --project scripts/train.jl path/to/settings.toml [--continue|--overwrite]`)

Runs the full pipeline: load data → derive/validate settings → create model →
train → save weights + settings → write outputs (plots, stats, `summary.toml`)
into `model_dir` (explicit in `[model_settings]`, or derived as
`training_output/<runid>_<model_name>` next to the TOML — see
[run info settings](run_info_settings.md)).

**If `model_dir` already exists**, `bin/train` refuses to run without an
explicit flag, rather than guessing what you want:

| Flag | Behaviour |
|---|---|
| *(none)* | Errors immediately, naming the existing directory. Nothing is touched. |
| `--continue` | Proceeds as normal. If a weights file (`params.jld2` or whatever `model_weights` points to) is present, it's loaded first and training continues from it. |
| `--overwrite` | Deletes any `params*` files in `model_dir` (weights and epoch checkpoints) first, then trains from scratch as if the directory were new. |

This only triggers when `model_dir` already exists — a first run into a new
directory needs no flag. Calling `train()` directly from Julia takes the same
choice as a keyword: `train(toml; on_existing_run=:continue)` (default
`:error`).

## Run inference

```bash
bin/predict path/to/settings.toml
```

(equivalently `pixi run julia --project scripts/predict.jl path/to/settings.toml`)

Loads a trained model's weights and settings from `model_dir` (see
`[model_settings]` in [settings reference](settings.md)) and runs inference,
writing whatever `[output_settings]` requests (see
[output settings](output_settings.md)) into the settings file's own
`model_dir` (or `predict_dir`, when set).

## Windows

`bin/train.bat` and `bin/predict.bat` are the Windows equivalents of
`bin/train`/`bin/predict`.

---

## Parameter sweeps

```bash
pixi run julia --project scripts/parameter_sweep.jl \
    [base.toml] [dotted.param.path] [v1,v2,v3] [nrepeats] [experiment] [--continue|--overwrite]
```

Trains one reference config's unmodified baseline plus one run per value in
`v1,v2,v3` (each repeated `nrepeats` times with a different seed), varying a
single dotted setting path (e.g. `model_settings.nlags`). All positional
arguments are optional and fall back to defaults defined at the top of the
script — edit those for a sweep you'll reuse, or override positionally for a
one-off.

Output goes to `sweeps/<experiment>/`:

- `<tag>/` — one `model_dir` per run (`baseline_rep<k>` and
  `<param>=<value>_rep<k>`), each a normal `train()` output directory
- `cfg_<tag>.toml` — the resolved settings file used for that run
- `results.csv` — one row per setting (baseline + each swept value), with
  the mean, std, and % reduction vs. baseline for `rmse_testing` and
  `rmse_storm_eunice_2022` across repeats

**If `sweeps/<experiment>/` already exists**, the script refuses to run
without a flag, same principle as `bin/train`:

| Flag | Behaviour |
|---|---|
| *(none)* | Errors immediately, naming the existing sweep directory. |
| `--continue` | Resumes it: any run tag whose `model_dir` already has a complete `summary.toml` is skipped; every other tag (never started, or interrupted partway through) is (re)trained **from scratch**, never warm-started. |
| `--overwrite` | Deletes `sweeps/<experiment>/` entirely and starts over. |

Sweep points are always trained from scratch rather than resumed mid-training
(even under `--continue`) — a value in the sweep that happened to warm-start
from a partial previous attempt wouldn't be a fair comparison against sibling
values trained fully from scratch.

---

## Smoke-testing everything

```bash
bash check_training_scripts.sh
```

Runs every example config in `examples/` through `bin/train --overwrite`
and (where a matching `predict_*.toml` exists) `bin/predict`, in parallel
when GNU parallel is available. Intended as an end-to-end sanity check after
code changes, not for real training runs — example configs use a low
`nepochs` for speed.
