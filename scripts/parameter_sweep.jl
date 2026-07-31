#!/usr/bin/env julia
# scripts/parameter_sweep.jl
#
# Vary ONE parameter of a reference training config over a list of values, N repeats
# each, plus the unmodified config as a baseline. Runs go into
# `sweeps/<experiment>/<setting>_rep<k>/`. Plots are kept by default (edit
# `plot_outputs_default` below to turn them off for speed on large sweeps);
# `write_series` stays off by default (bulky, rarely needed). After all runs, prints +
# writes `results.csv`: per setting the metric mean ± std across repeats (consistency)
# and the % RMSE reduction vs the unmodified baseline.
#
# Usage — edit the defaults below, or pass positionally:
#   pixi run julia --project scripts/parameter_sweep.jl \
#       [base.toml] [dotted.param.path] [v1,v2,v3] [nrepeats] [experiment] [--continue|--overwrite]
#
# If `sweeps/<experiment>/` already exists, the script refuses to run without one of:
#   --continue  — resume it: any run tag whose output dir already has a completed
#                 `summary.toml` is skipped; everything else (never started, or
#                 interrupted partway through) is retrained from scratch, so
#                 re-running the same command resumes an interrupted sweep without
#                 redoing finished runs.
#   --overwrite — delete `sweeps/<experiment>/` entirely and start over.
# Whether resuming makes sense (e.g. extending a sweep with more reps/values vs. stale
# runs from an earlier attempt) is a judgment call left to whoever passes `--continue`.

using AIHydroPoints, Random, Statistics, Printf, CSV, DataFrames

# ── configuration (edit defaults, or override via ARGS) ───────────────────────────
_parsenum(s) = something(tryparse(Int, String(s)), parse(Float64, String(s)))

# defaults
base_toml_default = "experiments/317stations/surge_5yr_BiLinearSurgeInteractionModel.toml"
param_path_default = ["model_settings", "nlags"]
values_default = Any[3*24, 4*24, 5*24, 6*24]   # 3,4,5,6 days
nrepeats_default = 1
experiment_default = "surge_317s_5yr_nlags48plus_sweep"
plot_outputs_default = false    # keep plots per run; set false for large sweeps (speed)
write_series_default = false   # write full output series per run; off by default (bulky)

continue_sweep  = "--continue" in ARGS
overwrite_sweep = "--overwrite" in ARGS
continue_sweep && overwrite_sweep &&
    error("Pass at most one of --continue / --overwrite.")
posargs = filter(a -> a ∉ ("--continue", "--overwrite"), ARGS)

base_toml  = length(posargs) >= 1 ? posargs[1] :
             base_toml_default
param_path = String.(length(posargs) >= 2 ? split(posargs[2], ".") :
             param_path_default)
values     = length(posargs) >= 3 ? _parsenum.(split(posargs[3], ",")) : values_default
nrepeats   = length(posargs) >= 4 ? parse(Int, posargs[4]) : nrepeats_default
experiment = length(posargs) >= 5 ? posargs[5] : experiment_default

const SWEEP_ROOT = "sweeps"
const METRICS    = ["rmse_testing", "rmse_storm_eunice_2022"]

pname     = param_path[end]
base_dir  = dirname(abspath(base_toml))
sweep_dir = joinpath(SWEEP_ROOT, experiment)
if isdir(sweep_dir)
    if overwrite_sweep
        rm(sweep_dir; recursive=true, force=true)
    elseif !continue_sweep
        error("sweep dir $sweep_dir already exists. Pass --continue to resume it " *
              "(skip finished runs, retrain unfinished ones), or --overwrite to " *
              "discard it and start over.")
    end
end
mkpath(sweep_dir)

# ── helpers ───────────────────────────────────────────────────────────────────────
function _setpath!(d, path, val)
    for k in path[1:end-1]; d = d[k]; end
    d[path[end]] = val
end

function _getpath(d, path)
    for k in path
        (d isa AbstractDict && haskey(d, k)) || return nothing
        d = d[k]
    end
    return d
end

"Keep the output entries (so summary RMSEs are still computed); toggle plots and
`write_series` per the `plot_outputs`/`write_series` flags."
function _apply_output_overrides!(cfg; plot_outputs, write_series)
    for e in get(get(cfg, "output_settings", Dict()), "outputs", Any[])
        for k in ("plot_timeseries", "plot_fft", "plot_scatter", "plot_stats",
                  "scatter_add_fit", "scatter_add_qq")
            e[k] = plot_outputs
        end
        e["write_series"] = write_series
        e["write_stats"] = true
    end
end

"`summary.toml` is the last file `train()` writes -- after all per-split stats/plots --
so its presence (and parseability, in case a kill landed mid-write) means `md` holds a
completed run, not one interrupted partway through."
function _run_complete(md)
    path = joinpath(md, "summary.toml")
    isfile(path) || return false
    try
        AIHydroPoints.toml_read(path)
        true
    catch
        false
    end
end

"Run one config: `value === nothing` is the unmodified baseline. Skips training (still
counts towards the run total) if `tag`'s output dir already completed -- lets a sweep
interrupted partway through be resumed by just re-running the same command."
function _run(tag, value)
    _run_counter[] += 1
    setting = value === nothing ? "$(pname) = $(base_val)  (unmodified baseline)" :
                                  "$(pname) = $(value)"
    md = abspath(joinpath(sweep_dir, tag))
    if _run_complete(md)
        @info @sprintf("[run %d/%d] %-24s | %s -- already complete, skipping",
                        _run_counter[], TOTAL_RUNS, tag, setting)
        return
    end
    retraining = isdir(md)
    @info @sprintf("[run %d/%d] %-24s | %s%s", _run_counter[], TOTAL_RUNS, tag, setting,
                    retraining ? " -- previous attempt incomplete, retraining from scratch" : "")
    cfg = AIHydroPoints.toml_read(base_toml)
    for f in cfg["data_settings"]["files"]              # make data paths absolute
        f["path"] = normpath(joinpath(base_dir, f["path"]))
    end
    value === nothing || _setpath!(cfg, param_path, value)
    _apply_output_overrides!(cfg; plot_outputs=plot_outputs_default, write_series=write_series_default)
    cfg["model_settings"]["model_dir"] = md
    tmp = joinpath(sweep_dir, "cfg_$tag.toml")
    AIHydroPoints.toml_write(tmp, cfg; overwrite=true)
    AIHydroPoints.train(tmp; on_existing_run=:overwrite)
end

function _metrics(tag)
    path = joinpath(sweep_dir, tag, "summary.toml")
    isfile(path) || return Dict(m => NaN for m in METRICS)
    s = read(path, String)
    Dict(m => (x = match(Regex("$(m)\\s*=\\s*([0-9.eE+-]+)"), s);
               x === nothing ? NaN : parse(Float64, x[1])) for m in METRICS)
end

_agg(tags) = Dict(m => (xs = filter(!isnan, [_metrics(t)[m] for t in tags]);
                        isempty(xs) ? (NaN, NaN) : (mean(xs), std(xs))) for m in METRICS)

# ── run: baseline (unmodified) + each value, × repeats ─────────────────────────────
const TOTAL_RUNS = nrepeats * (1 + length(values))    # baseline + values, × repeats
const _run_counter = Ref(0)
base_val = _getpath(AIHydroPoints.toml_read(base_toml), param_path)   # for the baseline log

@printf(">>> sweep '%s': vary %s over %s  (%d repeats + unmodified baseline)\n",
        experiment, join(param_path, "."), values, nrepeats)
for rep in 1:nrepeats
    Random.seed!(rep); _run("baseline_rep$rep", nothing)
end
for v in values, rep in 1:nrepeats
    Random.seed!(rep); _run("$(pname)=$(v)_rep$rep", v)
end

# ── collect + compare (% reduction vs the unmodified baseline) ─────────────────────
base_agg = _agg(["baseline_rep$rep" for rep in 1:nrepeats])

rows = DataFrame()
function _pushrow!(label, a)
    r = Dict{String, Any}("setting" => label)
    for m in METRICS
        mean_, std_ = a[m]; bm = base_agg[m][1]
        r["$(m)_mean"]    = round(mean_; digits=5)
        r["$(m)_std"]     = round(std_;  digits=5)
        r["$(m)_pct_red"] = (isnan(bm) || bm == 0) ? NaN : round(100 * (bm - mean_) / bm; digits=2)
    end
    push!(rows, r; cols = :union)
end
_pushrow!("baseline", base_agg)
for v in values
    _pushrow!("$(pname)=$(v)", _agg(["$(pname)=$(v)_rep$rep" for rep in 1:nrepeats]))
end

ordered = ["setting"]                               # readable, grouped column order
for m in METRICS; append!(ordered, ["$(m)_mean", "$(m)_std", "$(m)_pct_red"]); end
rows = rows[:, ordered]

CSV.write(joinpath(sweep_dir, "results.csv"), rows)
println("\n=== $experiment: mean±std over $nrepeats repeats; %_red vs unmodified baseline ===")
show(rows; allrows = true, allcols = true); println()
println("\nwrote ", joinpath(sweep_dir, "results.csv"))
