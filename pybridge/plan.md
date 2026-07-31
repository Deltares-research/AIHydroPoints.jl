# pybridge plan

## Environment notes

- Python is pinned to `<3.14` in `pixi.toml`. mlflow 3.14.0's tracking server
  crashes on Python 3.14 with `ImportError: cannot import name 'Traversable'
  from 'importlib.abc'` (upstream bug: https://github.com/mlflow/mlflow/issues/24155).
  Currently resolves to Python 3.13.14. **Recheck later** — once that issue is
  fixed upstream, try lifting the `<3.14` pin again.

## Design: mlflow training wrapper

Goal: wrap the existing Julia training pipeline (`bin/train settings.toml`)
so a run shows up in mlflow, without changing anything outside `pybridge/`.
No mlflow "Model"/registry — just params + artifacts + metrics on a run
(see discussion above on Model vs. artifact logging).

Three independent steps, mirroring how the pieces were built/tested so far:

1. **Before training** — parse `settings.toml` with the `toml` package.
   - Flatten `run_info`/`model_settings`/`train_settings`/`data_settings`
     into scalar key/value pairs and `mlflow.log_params(...)`.
   - Compute `model_dir` the same way Julia does (`input_processing.jl:112`):
     explicit `model_settings.model_dir` if set, else
     `training_output/<run_info.runid>_<model_settings.model_name>`,
     resolved relative to the toml's directory.
     **Caveat**: duplicates a one-line Julia convention in Python — if that
     derivation rule changes, this silently breaks. Acceptable for v1;
     revisit if it becomes a real papercut.
   - Experiment name = the toml's parent directory name (e.g. `317stations`),
     mirroring the grouping already used in `experiments/leaderboard.qmd`.

2. **During training** — run `bin/train <settings.toml>` as a non-blocking
   subprocess (`Popen`, not `run`), tee stdout/stderr to a log file, and
   concurrently poll `model_dir/losses.csv` for new rows (see format below),
   logging each as `mlflow.log_metric("train_rmse"/"val_rmse", value,
   step=epoch)` as it appears. Wait for the subprocess to finish. Non-zero
   exit → log the captured log as an artifact, mark the mlflow run `FAILED`,
   propagate the error.
   - Assumes the mlflow server is already running (`bin/mlflow-server-start`).
     The training wrapper does not start/stop the server itself.

3. **After training** — inspect `model_dir`:
   - Log `run_settings.toml`, `model_settings.toml`, `params.jld2` (or
     `params_best.jld2`), `losses.png`, `stats_<name>.csv` as run artifacts.
   - Parse `summary.toml` (written by `write_outputs`,
     `abstract_flux_model.jl:597-634`) and log its numeric fields —
     `n_params`, `train_time_s`, `rmse_<name>`, `predict_time_<name>_s` —
     as mlflow metrics. This is already a ready-made scalar summary; no CSV
     aggregation needed on the Python side for these.

### `losses.csv` format

`train_model!` returns `train_losses`/`val_losses`, but `train.jl` only
turns them into `losses.png` today — nothing machine-readable is persisted
during the run. Settled format for a `losses.csv` written (and flushed)
one row per epoch into `model_dir`:

```csv
epoch,train_rmse,val_rmse
1,0.8231,0.8450
2,0.7625,0.8102
3,0.7104,0.7803
4,0.6688,
```

- Header row, `csv.DictReader`-friendly (name-keyed, not position-keyed).
- `epoch`: 1-indexed, matches Julia's `for epoch in 1:nepochs` and the
  existing `epoch %d/%d` wording in the `@info` log message.
- `train_rmse`/`val_rmse`: named to match the values already computed in
  `abstract_flux_model.jl` (not a generic "loss").
- `val_rmse` blank when the run has no validation split — fixed 3-column
  shape always, rather than a variable column count.
- No timestamp column — mlflow assigns its own timestamp when the poller
  calls `log_metric`.
- Julia writer must `flush()` (or close/reopen) after each row so the
  Python poller sees it promptly. Python poller keeps the file open,
  tracks a row-count bookmark, and only treats fully-written rows as new
  (standard `tail -f` semantics) — no per-check close/reopen needed on the
  read side.

**Implemented.** `abstract_flux_model.jl`'s epoch loop rewrites (not
appends) `<model_dir>/losses.csv` every epoch from the in-memory
`train_losses`/`val_losses` vectors, via `CSV.write(..., DataFrame(epoch=...,
train_rmse=round.(...; digits=6), val_rmse=has_val ? round.(...; digits=6) :
missing))`. `CSV.write` opens/writes/closes atomically each call, so there's
no persistent file handle to manage across early stopping's `break`. Values
rounded to 6 digits, matching `summary.toml`'s `rmse_*` convention.

**Stale-file race (found and fixed during testing):** a leftover
`losses.csv` from a *previous* run in the same `model_dir` could get read by
`mlflow_train.py`'s poller before the new Julia process even reached its
training loop (Julia takes several seconds to start up/precompile first),
poisoning the poller's row-count bookmark so the new run's real rows were
silently never logged once written (confirmed empirically: the previous
run's stale values showed up in mlflow's metric history instead of the new
run's). Fixed on both sides: `train_model!` now deletes any existing
`losses.csv` at the very start of the function (before the loop), and
`mlflow_train.py`'s `run_training()` also deletes it before even launching
the subprocess — the Python-side delete is the one that actually closes the
race (it runs before the poller thread starts), the Julia-side delete is
defense-in-depth for anyone calling `train_model!` directly. Verified with a
real run: `losses.csv` on disk and mlflow's `metrics/get-history` now match
exactly, step for step.

## Design: mlflow parameter sweep logging (settled)

Goal: get parameter sweeps visible in mlflow too — comparable/filterable in
the UI the same way single training runs already are.

Earlier sketch considered and dropped: `pybridge/mlflow_sweep.py` launching
`parameter_sweep.jl` as a subprocess and polling `sweeps/<experiment>/*/`
for newly-complete points, using mlflow's nested-runs feature (one parent
run per sweep, one child run per point). Dropped in favor of the simpler
design below, which is more deterministic (no polling/race concerns) and
reuses `mlflow_train.py` as-is per point rather than needing new nested-run
plumbing.

**Settled architecture**: `pybridge/mlflow_sweep.py` reimplements
`parameter_sweep.jl`'s sweep loop directly in Python — same CLI shape
(`base_toml`, `dotted.param.path`, `values`, `nrepeats`, `experiment`,
`--continue`/`--overwrite`), same tag naming (`baseline_rep<k>`,
`<param>=<value>_rep<k>`), same skip-if-already-complete check (a point's
`model_dir` has a valid, parseable `summary.toml`). For each point that
needs (re)training, it calls `bin/python mlflow_train.py <cfg_tag.toml>` as
a subprocess — reusing all of `mlflow_train.py`'s existing training +
logging machinery unmodified, rather than reimplementing it. Only the sweep
*orchestration* is duplicated between the two languages, not the actual
train/log logic.

**Known tradeoff, accepted**: sweep orchestration logic now exists in two
places (Julia and Python). If `parameter_sweep.jl` changes (new flag, new
tag format, new default), this won't follow automatically — has to be
mirrored by hand.

Settled decisions:

- **Output location: exactly `sweeps/<experiment>/`**, same as
  `parameter_sweep.jl` — not a separate `pybridge/`-local directory. Since
  every point, regardless of which tool triggered it, ultimately calls the
  same `AIHydroPoints.train()` pipeline (Python's route:
  `mlflow_sweep.py` → `mlflow_train.py` → `bin/train` → `train()`; Julia's:
  direct), output is genuinely format-identical, not just similarly shaped
  — a sweep can be resumed with `--continue` via either tool interchangeably.
- **`results.csv`**: also written to `sweeps/<experiment>/results.csv`, same
  computation (mean/std/% reduction vs. baseline across repeats), same
  approach (regenerated from whatever points are complete on disk) — keeps
  the two tools' output substitutable, not just their inputs.
- **mlflow experiment grouping: explicit, not inferred.** `mlflow_sweep.py`
  takes `experiment` as an explicit argument (mirrors Julia's positional arg
  exactly) and sets `MLFLOW_EXPERIMENT_NAME=<experiment>` in the environment
  when invoking each point's `mlflow_train.py` subprocess call.
- **Small `mlflow_train.py` change needed**: only fall back to its current
  `settings_toml.parent.name`-derived experiment name when
  `MLFLOW_EXPERIMENT_NAME` isn't already set in the environment —
  `mlflow.start_run()` picks up that env var on its own otherwise (it's
  mlflow's own standard mechanism for this). Zero change to standalone
  (non-sweep) behavior, since the env var is simply unset in that case.
- **Per-point skip logic** mirrors `parameter_sweep.jl`'s `_run_complete`:
  check for a valid, parseable `summary.toml` in the point's `model_dir`
  before deciding whether to invoke `mlflow_train.py` for that point at all.
- **When a point *is* run**, `mlflow_train.py` is called completely
  unmodified for the training/logging part — it already always passes
  `--overwrite` to `bin/train`, which matches `parameter_sweep.jl`'s own
  `on_existing_run=:overwrite` per point (never warm-start a sweep point,
  since that would bias comparisons across the sweep).

**Implemented** as `pybridge/mlflow_sweep.py`, plus the small
`MLFLOW_EXPERIMENT_NAME` opt-out added to `mlflow_train.py`. Verified against
a real (small, fast) sweep: fresh `--overwrite` run, dotted-path override
confirmed both in the generated `cfg_<tag>.toml` and in the logged mlflow
params, `--continue` correctly skips already-complete points and only
(re)trains new/incomplete ones, no-flag correctly refuses against an
existing sweep dir, all points landed in one mlflow experiment via
`MLFLOW_EXPERIMENT_NAME`, and `results.csv` matches `parameter_sweep.jl`'s
column layout and values exactly (including `"NaN"` formatting for
single-repeat runs' std).

**Known limitations, not fixed:**
- **No cross-repeat reproducibility.** `parameter_sweep.jl` calls
  `Random.seed!(rep)` before each point within one long-lived Julia process.
  Here, each point is a fresh Julia subprocess (via `mlflow_train.py` →
  `bin/train`), and nothing seeds it — `bin/train` has no seed option today.
  Repeats are therefore not reproducible run-to-run via this tool, unlike
  via `parameter_sweep.jl`. Fixing this would mean adding a seed CLI option
  to `bin/train`/`scripts/train.jl`/`train()`, a `src/`/`scripts/` change
  outside today's scope.
- **Every point in a sweep gets the same mlflow run name.** `run_name` is
  set from the toml's `run_info.runid` (see the earlier "runid as a
  column" fix), and — matching `parameter_sweep.jl` exactly — neither tool
  touches `run_info` when generating a point's config, only
  `model_settings.model_dir` (and the swept param). So every point in a
  sweep shares one run name in the mlflow UI (e.g. all show as `surge_1yr`),
  distinguishable only via params (e.g. `model_settings.nlags`) or the tag
  in `model_dir`'s path — not at a glance the way single runs are. Worth a
  follow-up decision: derive each point's `run_name` from its tag instead
  (e.g. `surge_1yr/nlags=384_rep1`)?

## Follow-up tasks

- ~~**Explicit `--continue`/`--overwrite` for existing runs.**~~ **Done.**
  `train()` (`src/train.jl`) previously silently warm-started from
  `<model_dir>/params.jld2` whenever it already existed — easy to trigger by
  accident (e.g. re-running the same settings.toml, as happened repeatedly
  while testing the mlflow wrapper). Now takes `on_existing_run::Symbol`
  (`:error` default, `:continue`, `:overwrite`), checked via `isdir(model_dir)`
  *before* any writes — any pre-existing `model_dir` blocks by default, not
  just one with a weights file already in it. `:overwrite` deletes all
  `params*` files (weights + epoch checkpoints) before training. Plumbed
  through as CLI flags: `bin/train <settings.toml> [--continue|--overwrite]`,
  `scripts/train.jl` parses them too. `scripts/parameter_sweep.jl`'s old
  single `--force` flag was split into sweep-level `--continue` (skip
  finished points, retrain unfinished ones from scratch — never warm-start a
  sweep point, since that would bias comparisons across the sweep) and
  `--overwrite` (delete the whole `sweep_dir` and start over); it now passes
  `on_existing_run=:overwrite` per-point instead of its own manual
  `params*`-deletion loop. `mlflow_train.py` and `check_training_scripts.sh`
  both updated to pass `--overwrite`, since they need every (re-)run to be
  independent/comparable rather than continuing from whatever was last
  there. Full test suite (658 tests) and `check_training_scripts.sh` (12/12)
  both pass after the change; `test/test_pipeline.jl` updated to pass
  `on_existing_run=:overwrite` explicitly since its `model_dir` persists
  across test runs.

- **Multi-machine tracking server.** Currently `bin/mlflow-server-start`
  binds to localhost only, so only the machine running the server can log
  to it. To let multiple training machines log to one shared server:
  pass `--host 0.0.0.0` and configure `--allowed-hosts`/CORS (the server
  already warns about this — see its startup log), then point each other
  machine's `MLFLOW_TRACKING_URI` at `http://<server-host>:5000`. SQLite
  backend store + local-disk artifact store on the server host remain fine
  even for remote clients, since everything goes through the server's REST
  API (no shared filesystem needed) — no need to move to Postgres/S3 for
  this. Caveat: mlflow's tracking server has no built-in auth, so only bind
  beyond `127.0.0.1` on a trusted LAN/VPN, not an open network, unless auth
  (basic-auth plugin or reverse proxy) is added first. Untested — revisit
  once there's a second machine to actually try it against.
