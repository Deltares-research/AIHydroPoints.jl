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

For now: `pybridge/example_losses.csv` is a dummy fixture (fake rows in
this exact format) to build and test the polling/parsing logic against.
**Follow-up (needs separate go-ahead, touches `src/` outside `pybridge/`)**:
teach `train.jl` to write a real `losses.csv` incrementally during training
so the wrapper logs actual live per-epoch RMSE curves.

## Follow-up tasks

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
