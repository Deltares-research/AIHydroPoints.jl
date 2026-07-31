"""Wrap `bin/train <settings.toml>` (the Julia training pipeline) as an mlflow run.

Usage:
    bin/python mlflow_train.py <settings.toml>

Assumes the mlflow tracking server is already running (bin/mlflow-server-start).
"""

import csv
import os
import subprocess
import sys
import threading
from pathlib import Path

import mlflow
import toml

REPO_DIR = Path(__file__).resolve().parent.parent
PYBRIDGE_DIR = Path(__file__).resolve().parent
RUN_LOG_DIR = PYBRIDGE_DIR / "run"


def flatten(d, prefix=""):
    """Recursively flatten a nested dict into a single-level dict.

    Nested keys are joined with ".", e.g. {"a": {"b": 1}} -> {"a.b": 1}.
    Used to turn a settings.toml section into scalar key/value pairs
    suitable for mlflow.log_params(), which does not accept nested dicts.

    `prefix` is the dotted key prefix to prepend (used for the recursive
    calls); pass a section name here when flattening a subsection, e.g.
    flatten(settings["model_settings"], "model_settings").
    """
    items = {}
    for key, value in d.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            items.update(flatten(value, full_key))
        else:
            items[full_key] = value
    return items


def resolve_model_dir(settings, toml_dir):
    """Compute the absolute model_dir for a settings dict, mirroring Julia.

    Julia's train() (via validate_and_augment_settings!, see
    src/input_processing.jl:108-113) uses model_settings["model_dir"] if
    given, otherwise derives training_output/<runid>_<model_name> relative
    to the settings.toml's own directory. This duplicates that one-line
    convention in Python so we can locate a run's output directory before
    (and independently of) invoking Julia. `toml_dir` is the directory
    containing the settings.toml (relative paths are resolved against it,
    same as Julia does).
    """
    model_settings = settings["model_settings"]
    if "model_dir" in model_settings:
        model_dir = Path(model_settings["model_dir"])
    else:
        runid = settings["run_info"]["runid"]
        model_name = model_settings["model_name"]
        model_dir = Path("training_output") / f"{runid}_{model_name}"
    if not model_dir.is_absolute():
        model_dir = toml_dir / model_dir
    return model_dir


def is_git_dirty(repo_dir):
    """Return True if `repo_dir` has uncommitted changes (tracked or untracked).

    Used to tag mlflow runs with whether the working tree was clean, since
    results depend on the exact src/ code, not just the settings.toml, and
    mlflow's automatic git tag only records the last commit, not dirty state.
    """
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo_dir, capture_output=True, text=True
    )
    return bool(result.stdout.strip())


def poll_losses_csv(losses_csv, run_id, stop_event):
    """Poll `losses_csv` for new rows and log them to mlflow as step metrics.

    Intended to run in a background thread alongside the training
    subprocess (see run_training()) so per-epoch train/val RMSE show up in
    the mlflow UI while training is still in progress, rather than only
    after it finishes. `run_id` is passed explicitly to mlflow.log_metric()
    (rather than relying on the ambient active run) because this runs on a
    separate thread from the one that opened the run.

    Call `stop_event.set()` to make the loop do one final read-and-log pass
    (to pick up any rows written just before the caller stops it) and then
    return; the thread does not exit on its own.
    """
    seen = 0
    while True:
        # Re-read the whole file each poll and skip rows already logged
        # (tracked via `seen`), rather than seeking by byte offset — the
        # file is small, and csv.DictReader naturally ignores a not-yet-
        # newline-terminated trailing row, so a partial last write is safe.
        if losses_csv.exists():
            with open(losses_csv, newline="") as f:
                rows = list(csv.DictReader(f))
            for row in rows[seen:]:
                epoch = int(row["epoch"])
                if row.get("train_rmse"):
                    mlflow.log_metric("train_rmse", float(row["train_rmse"]), step=epoch, run_id=run_id)
                if row.get("val_rmse"):
                    mlflow.log_metric("val_rmse", float(row["val_rmse"]), step=epoch, run_id=run_id)
            seen = len(rows)
        if stop_event.is_set():
            break
        stop_event.wait(1.0)


def run_training(settings_toml_arg, run_id, log_file):
    """Run `bin/train <settings_toml_arg>` to completion and return its result.

    Launches the Julia training pipeline as a subprocess, tees its merged
    stdout/stderr to `log_file` as it runs, and concurrently polls for
    `model_dir/losses.csv` in a background thread (see poll_losses_csv())
    so live per-epoch metrics reach mlflow during training, not just after.

    Returns (returncode, model_dir): the subprocess exit code, and the
    resolved model_dir the caller should look in for training outputs.
    """
    settings_toml = Path(settings_toml_arg).resolve()
    toml_dir = settings_toml.parent
    settings = toml.load(settings_toml)
    model_dir = resolve_model_dir(settings, toml_dir)
    losses_csv = model_dir / "losses.csv"

    # Julia deletes losses.csv itself at the start of train_model!, but that
    # happens only after several seconds of startup/precompilation — too
    # late to stop the poller below from reading a *previous* run's
    # leftover file the moment it starts and bookmarking past it, silently
    # dropping this run's real rows once they arrive (seen empirically).
    # Clearing it here, before the subprocess/poller even start, closes
    # that window from the reader's side too.
    losses_csv.unlink(missing_ok=True)

    # bufsize=1 (line-buffered) + text=True so `for line in proc.stdout`
    # yields lines as they're written, not only once the process exits.
    #
    # --overwrite: bin/train now refuses to run against an existing model_dir
    # unless told --continue or --overwrite (added to guard against
    # accidentally warm-starting a run). Each mlflow run should be an
    # independent, comparable training run, not a continuation of whatever
    # was last logged against the same settings.toml, so always overwrite.
    proc = subprocess.Popen(
        [str(REPO_DIR / "bin" / "train"), str(settings_toml), "--overwrite"],
        cwd=REPO_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Start the losses.csv poller before consuming stdout below, so it runs
    # concurrently with (not after) the blocking read loop.
    stop_event = threading.Event()
    poller = threading.Thread(target=poll_losses_csv, args=(losses_csv, run_id, stop_event))
    poller.start()

    with open(log_file, "w") as f:
        for line in proc.stdout:
            f.write(line)
            f.flush()

    # proc.stdout hits EOF (ending the loop above) once the process exits,
    # so proc.wait() here just reaps it and captures the exit code.
    proc.wait()
    stop_event.set()
    poller.join()

    return proc.returncode, model_dir


def log_artifacts(model_dir, log_file):
    """Log the standard set of training-output files as mlflow run artifacts.

    Covers everything `train()` writes into model_dir that's useful to keep
    with the run (resolved settings, weights, loss plot, per-split stats,
    the machine-readable summary), plus the captured training log. Missing
    files are skipped rather than erroring, since not every settings.toml
    produces every file (e.g. no validation split -> no params_best.jld2).
    """
    for name in ["run_settings.toml", "model_settings.toml", "params.jld2",
                 "params_best.jld2", "losses.png", "losses.csv", "summary.toml"]:
        path = model_dir / name
        if path.exists():
            mlflow.log_artifact(str(path))
    for path in model_dir.glob("stats_*.csv"):
        mlflow.log_artifact(str(path))
    mlflow.log_artifact(str(log_file))


def log_summary_metrics(model_dir):
    """Log the numeric fields of model_dir/summary.toml as mlflow metrics.

    summary.toml is written by Julia's write_outputs() (see
    src/models/abstract_flux_model.jl:597-634) and already contains a
    ready-made scalar summary of the run (n_params, train_time_s,
    rmse_<split>, predict_time_<split>_s, ...) — this just relays every
    numeric field as-is, without recomputing anything. Non-numeric fields
    (runid, description, model_name, ...) are skipped since they duplicate
    what's already logged as params from the input settings.toml.
    """
    summary_path = model_dir / "summary.toml"
    if not summary_path.exists():
        return
    summary = toml.load(summary_path)
    for key, value in summary.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            mlflow.log_metric(key, value)


def main():
    """Parse CLI args and run one settings.toml as a single mlflow run.

    See module docstring for usage. Log layout follows pybridge/plan.md's
    "Design: mlflow training wrapper" section: params from the input toml
    are logged before training starts, the training subprocess runs with
    live loss polling, and artifacts/summary metrics are logged after it
    finishes (or the run is marked FAILED and the exit code is propagated
    if the subprocess itself failed).
    """
    if len(sys.argv) != 2:
        print("Usage: mlflow_train.py <settings.toml>", file=sys.stderr)
        sys.exit(1)

    settings_toml = Path(sys.argv[1]).resolve()
    settings = toml.load(settings_toml)

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    # One mlflow experiment per settings.toml directory (e.g. "5stations",
    # "317stations"), mirroring the grouping already used by
    # experiments/leaderboard.qmd -- unless a caller (e.g. mlflow_sweep.py,
    # to group every point of one sweep together) has already set
    # MLFLOW_EXPERIMENT_NAME, mlflow's own standard override for this;
    # mlflow.start_run() picks that up on its own, so nothing further to do
    # here in that case.
    if "MLFLOW_EXPERIMENT_NAME" not in os.environ:
        mlflow.set_experiment(settings_toml.parent.name)

    # Flatten just the sections worth logging as params. data_settings.files
    # (the list of input/target file entries) is deliberately excluded: it's
    # a list of dicts, not scalars, and would produce noisy/oversized param
    # values — the full data config is still captured via the
    # run_settings.toml artifact logged after training.
    params = {}
    params.update(flatten(settings.get("run_info", {}), "run_info"))
    params.update(flatten(settings.get("model_settings", {}), "model_settings"))
    params.update(flatten(settings.get("train_settings", {}), "train_settings"))
    params.update(flatten(settings.get("data_settings", {}).get("model_io", {}), "data_settings.model_io"))

    RUN_LOG_DIR.mkdir(exist_ok=True)

    # run_name surfaces as the default identifying column in the mlflow UI
    # (replacing an auto-generated name like "resilient-crow-840"), unlike
    # params, which are only shown in the runs table once manually toggled
    # on via its column selector. Defaults to the toml's own runid, but a
    # caller (e.g. mlflow_sweep.py, to give each sweep point a distinct,
    # informative name instead of every point sharing the base runid) can
    # override it via MLFLOW_RUN_NAME -- unlike MLFLOW_EXPERIMENT_NAME this
    # isn't an mlflow-recognized variable, just our own convention, since
    # mlflow has no automatic env-var pickup for run_name.
    run_name = os.environ.get("MLFLOW_RUN_NAME", settings["run_info"]["runid"])
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.set_tags({
            "model_name": settings["model_settings"]["model_name"],
            "git_dirty": str(is_git_dirty(REPO_DIR)).lower(),
        })

        log_file = RUN_LOG_DIR / f"train-{run.info.run_id}.log"
        returncode, model_dir = run_training(settings_toml, run.info.run_id, log_file)

        if returncode != 0:
            # model_dir may not contain a full/valid output set on failure,
            # so just log the captured log and stop rather than attempting
            # log_artifacts()/log_summary_metrics() against a partial run.
            mlflow.log_artifact(str(log_file))
            mlflow.end_run(status="FAILED")
            print(f"Training failed (exit {returncode}); see {log_file}", file=sys.stderr)
            sys.exit(returncode)

        log_artifacts(model_dir, log_file)
        log_summary_metrics(model_dir)

        print(f"Logged run {run.info.run_id} to {tracking_uri}")


if __name__ == "__main__":
    main()
