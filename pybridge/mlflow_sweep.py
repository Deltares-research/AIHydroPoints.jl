"""Vary ONE parameter of a reference training config over a list of values, N repeats
each, plus the unmodified config as a baseline -- logging every point to mlflow.

This is a Python reimplementation of scripts/parameter_sweep.jl's orchestration
(same CLI shape, same tag naming, same sweeps/<experiment>/ output layout, same
results.csv), so a sweep is resumable interchangeably by either tool. Only the
orchestration is duplicated -- each point's actual training + mlflow logging is
delegated to mlflow_train.py, run unmodified as a subprocess, so nothing about
how a run is logged lives in two places.

Usage:
    bin/python mlflow_sweep.py \\
        [base.toml] [dotted.param.path] [v1,v2,v3] [nrepeats] [experiment] [--continue|--overwrite]

If sweeps/<experiment>/ already exists, this refuses to run without one of:
  --continue  -- resume it: any run tag whose output dir already has a completed
                 summary.toml is skipped; everything else (never started, or
                 interrupted partway through) is retrained from scratch.
  --overwrite -- delete sweeps/<experiment>/ entirely and start over.

Known gap vs. parameter_sweep.jl: Julia's version calls Random.seed!(rep) before
each point within one long-lived process, so repeats are reproducible. Here, each
point is a fresh Julia subprocess (via mlflow_train.py -> bin/train), and nothing
seeds it explicitly -- bin/train has no seed option today. Repeats are therefore
not reproducible run-to-run via this tool, unlike via parameter_sweep.jl.
"""

import csv
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

import toml

REPO_DIR = Path(__file__).resolve().parent.parent
PYBRIDGE_DIR = Path(__file__).resolve().parent

SWEEP_ROOT = REPO_DIR / "sweeps"
METRICS = ["rmse_testing", "rmse_storm_eunice_2022"]

# defaults, mirroring parameter_sweep.jl's
BASE_TOML_DEFAULT = "experiments/5stations/surge_5yr_LinearSurgeModel.toml"
PARAM_PATH_DEFAULT = ["model_settings", "nlags"]
VALUES_DEFAULT = [24, 2*24, 3 * 24, 4 * 24, 5 * 24]  # 1,2,3,4,5 days
NREPEATS_DEFAULT = 3
EXPERIMENT_DEFAULT = "surge_5s_5yr_nlags_sweep"
PLOT_OUTPUTS_DEFAULT = False
WRITE_SERIES_DEFAULT = False


def _parsenum(s):
    """Parse a CLI value as an int if possible, else a float (mirrors Julia's
    tryparse(Int, ...) / parse(Float64, ...) fallback)."""
    try:
        return int(s)
    except ValueError:
        return float(s)


def _setpath(d, path, val):
    """Set a value at a dotted path (list of keys) inside a nested dict, in place."""
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = val


def _getpath(d, path):
    """Read a value at a dotted path (list of keys) inside a nested dict, or None
    if any key along the way is missing."""
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _apply_output_overrides(cfg, plot_outputs, write_series):
    """Keep the output entries (so summary RMSEs are still computed); toggle plots
    and write_series per plot_outputs/write_series, matching
    parameter_sweep.jl's _apply_output_overrides!."""
    for entry in cfg.get("output_settings", {}).get("outputs", []):
        for key in ("plot_timeseries", "plot_fft", "plot_scatter", "plot_stats",
                    "scatter_add_fit", "scatter_add_qq"):
            entry[key] = plot_outputs
        entry["write_series"] = write_series
        entry["write_stats"] = True


def _run_complete(model_dir):
    """True if model_dir holds a completed run: summary.toml exists and parses.
    summary.toml is the last file train() writes, so its presence (and
    parseability, in case a kill landed mid-write) means this is a finished run,
    not one interrupted partway through -- mirrors parameter_sweep.jl's
    _run_complete."""
    path = model_dir / "summary.toml"
    if not path.is_file():
        return False
    try:
        toml.load(path)
        return True
    except Exception:
        return False


def _run(sweep_dir, base_toml, base_dir, param_path, pname, base_val, experiment,
         run_counter, total_runs, tag, value):
    """Run one sweep point: value is None for the unmodified baseline. Skips
    training (still counts towards the run total) if tag's output dir already
    completed -- lets a sweep interrupted partway through be resumed by just
    re-running the same command. Otherwise generates that point's resolved
    settings.toml (same as parameter_sweep.jl's _run) and hands it to
    mlflow_train.py, which runs bin/train and logs the result to mlflow --
    reused unmodified, so the actual train/log logic isn't duplicated here."""
    run_counter[0] += 1
    setting = (f"{pname} = {base_val}  (unmodified baseline)" if value is None
               else f"{pname} = {value}")
    model_dir = (sweep_dir / tag).resolve()

    if _run_complete(model_dir):
        print(f"[run {run_counter[0]}/{total_runs}] {tag:<24} | {setting} "
              f"-- already complete, skipping")
        return

    retraining = model_dir.is_dir()
    suffix = " -- previous attempt incomplete, retraining from scratch" if retraining else ""
    print(f"[run {run_counter[0]}/{total_runs}] {tag:<24} | {setting}{suffix}")

    cfg = toml.load(base_toml)
    for f in cfg["data_settings"]["files"]:              # make data paths absolute
        f["path"] = os.path.normpath(os.path.join(base_dir, f["path"]))
    if value is not None:
        _setpath(cfg, param_path, value)
    _apply_output_overrides(cfg, PLOT_OUTPUTS_DEFAULT, WRITE_SERIES_DEFAULT)
    cfg["model_settings"]["model_dir"] = str(model_dir)

    tmp = sweep_dir / f"cfg_{tag}.toml"
    with open(tmp, "w") as f:
        toml.dump(cfg, f)

    # MLFLOW_EXPERIMENT_NAME groups every point of this sweep into one mlflow
    # experiment (see mlflow_train.py's set_experiment logic); MLFLOW_RUN_NAME
    # gives each point a distinct, informative run name matching its
    # sweeps/<experiment>/<tag> folder name, instead of every point in the
    # sweep otherwise sharing the base toml's runid. mlflow_train.py itself
    # already always passes --overwrite to bin/train, matching
    # parameter_sweep.jl's on_existing_run=:overwrite per point (never
    # warm-start a sweep point -- that would bias comparisons across the sweep).
    env = os.environ.copy()
    env["MLFLOW_EXPERIMENT_NAME"] = experiment
    env["MLFLOW_RUN_NAME"] = f"{experiment}/{tag}"
    result = subprocess.run(
        [sys.executable, str(PYBRIDGE_DIR / "mlflow_train.py"), str(tmp)],
        cwd=REPO_DIR, env=env,
    )
    if result.returncode != 0:
        print(f"Sweep aborted: point {tag} failed (exit {result.returncode}).",
              file=sys.stderr)
        sys.exit(result.returncode)


def _metrics(sweep_dir, tag):
    """Read this point's METRICS from its summary.toml, NaN for any missing
    (including a point with no summary.toml at all) -- mirrors
    parameter_sweep.jl's _metrics."""
    path = sweep_dir / tag / "summary.toml"
    summary = {}
    if path.is_file():
        try:
            summary = toml.load(path)
        except Exception:
            summary = {}
    return {m: float(summary[m]) if m in summary else float("nan") for m in METRICS}


def _agg(sweep_dir, tags):
    """Mean and (sample) std of each metric across tags, skipping NaNs -- mirrors
    parameter_sweep.jl's _agg. A single non-NaN value gives NaN std (matching
    Julia's std of a 1-element vector), not an error."""
    result = {}
    for m in METRICS:
        xs = [v for t in tags for v in [_metrics(sweep_dir, t)[m]] if not math.isnan(v)]
        if not xs:
            result[m] = (float("nan"), float("nan"))
        else:
            mean_ = statistics.mean(xs)
            std_ = statistics.stdev(xs) if len(xs) > 1 else float("nan")
            result[m] = (mean_, std_)
    return result


def _round_or_nan(x, digits):
    return round(x, digits) if not math.isnan(x) else float("nan")


def _write_results_csv(sweep_dir, pname, values, nrepeats):
    """Collect + compare (% reduction vs. the unmodified baseline) and write
    results.csv -- mirrors parameter_sweep.jl's final section exactly (same
    columns, same rounding), so it's readable/regeneratable by either tool."""
    base_agg = _agg(sweep_dir, [f"baseline_rep{rep}" for rep in range(1, nrepeats + 1)])

    rows = []

    def _pushrow(label, agg):
        row = {"setting": label}
        for m in METRICS:
            mean_, std_ = agg[m]
            baseline_mean = base_agg[m][0]
            row[f"{m}_mean"] = _round_or_nan(mean_, 5)
            row[f"{m}_std"] = _round_or_nan(std_, 5)
            if math.isnan(baseline_mean) or baseline_mean == 0:
                row[f"{m}_pct_red"] = float("nan")
            else:
                row[f"{m}_pct_red"] = round(100 * (baseline_mean - mean_) / baseline_mean, 2)
        rows.append(row)

    _pushrow("baseline", base_agg)
    for v in values:
        tags = [f"{pname}={v}_rep{rep}" for rep in range(1, nrepeats + 1)]
        _pushrow(f"{pname}={v}", _agg(sweep_dir, tags))

    ordered = ["setting"]
    for m in METRICS:
        ordered += [f"{m}_mean", f"{m}_std", f"{m}_pct_red"]

    # "NaN" (not Python's lowercase "nan") to match Julia's CSV.jl output, so
    # results.csv reads the same regardless of which tool wrote it.
    with open(sweep_dir / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("NaN" if isinstance(v, float) and math.isnan(v) else v)
                              for k, v in row.items()})


def main():
    # Line-buffer stdout even when piped (not a tty), so this script's own
    # print()s interleave correctly with each mlflow_train.py subprocess's
    # output (which inherits our stdout fd directly, unbuffered from our
    # side) instead of all appearing after every subprocess has finished.
    sys.stdout.reconfigure(line_buffering=True)

    continue_flag = "--continue" in sys.argv[1:]
    overwrite_flag = "--overwrite" in sys.argv[1:]
    if continue_flag and overwrite_flag:
        print("Pass at most one of --continue / --overwrite.", file=sys.stderr)
        sys.exit(1)
    posargs = [a for a in sys.argv[1:] if a not in ("--continue", "--overwrite")]

    # A user-supplied base_toml stays CWD-relative (normal CLI convention,
    # matching mlflow_train.py's own positional arg); the *default* is
    # anchored to REPO_DIR instead, since it's meaningless relative to
    # wherever this script happens to be invoked from (e.g. pybridge/).
    base_toml = posargs[0] if len(posargs) >= 1 else str(REPO_DIR / BASE_TOML_DEFAULT)
    param_path = posargs[1].split(".") if len(posargs) >= 2 else PARAM_PATH_DEFAULT
    values = ([_parsenum(v) for v in posargs[2].split(",")] if len(posargs) >= 3
              else VALUES_DEFAULT)
    nrepeats = int(posargs[3]) if len(posargs) >= 4 else NREPEATS_DEFAULT
    experiment = posargs[4] if len(posargs) >= 5 else EXPERIMENT_DEFAULT

    pname = param_path[-1]
    base_dir = Path(base_toml).resolve().parent
    sweep_dir = SWEEP_ROOT / experiment

    if sweep_dir.is_dir():
        if overwrite_flag:
            import shutil
            shutil.rmtree(sweep_dir)
        elif not continue_flag:
            print(f"sweep dir {sweep_dir} already exists. Pass --continue to "
                  f"resume it (skip finished runs, retrain unfinished ones), or "
                  f"--overwrite to discard it and start over.", file=sys.stderr)
            sys.exit(1)
    sweep_dir.mkdir(parents=True, exist_ok=True)

    base_val = _getpath(toml.load(base_toml), param_path)   # for the baseline log
    total_runs = nrepeats * (1 + len(values))
    run_counter = [0]   # list, not int, so _run can mutate it in place

    print(f">>> sweep '{experiment}': vary {'.'.join(param_path)} over {values}  "
          f"({nrepeats} repeats + unmodified baseline)")

    for rep in range(1, nrepeats + 1):
        _run(sweep_dir, base_toml, base_dir, param_path, pname, base_val, experiment,
             run_counter, total_runs, f"baseline_rep{rep}", None)
    for v in values:
        for rep in range(1, nrepeats + 1):
            _run(sweep_dir, base_toml, base_dir, param_path, pname, base_val, experiment,
                 run_counter, total_runs, f"{pname}={v}_rep{rep}", v)

    _write_results_csv(sweep_dir, pname, values, nrepeats)


if __name__ == "__main__":
    main()
