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
#       [base.toml] [dotted.param.path] [v1,v2,v3] [nrepeats] [experiment]

using AIHydroPoints, Random, Statistics, Printf, CSV, DataFrames

# ── configuration (edit defaults, or override via ARGS) ───────────────────────────
_parsenum(s) = something(tryparse(Int, String(s)), parse(Float64, String(s)))

# defaults
base_toml_default = "experiments/317stations/surge_5yr_BiLinearSurgeInteractionModel.toml"
param_path_default = ["train_settings", "batch_size"]
values_default = Any[32, 128, 256, 512]
nrepeats_default = 1
experiment_default = "surge_317s_5yr_batch_size_sweep"
plot_outputs_default = true    # keep plots per run; set false for large sweeps (speed)
write_series_default = false   # write full output series per run; off by default (bulky)

base_toml  = length(ARGS) >= 1 ? ARGS[1] :
             base_toml_default
param_path = String.(length(ARGS) >= 2 ? split(ARGS[2], ".") :
             param_path_default)
values     = length(ARGS) >= 3 ? _parsenum.(split(ARGS[3], ",")) : values_default
nrepeats   = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : nrepeats_default
experiment = length(ARGS) >= 5 ? ARGS[5] : experiment_default

const SWEEP_ROOT = "sweeps"
const METRICS    = ["rmse_testing", "rmse_storm_eunice_2022"]

pname     = param_path[end]
base_dir  = dirname(abspath(base_toml))
sweep_dir = joinpath(SWEEP_ROOT, experiment)
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

"Run one config: `value === nothing` is the unmodified baseline."
function _run(tag, value)
    _run_counter[] += 1
    setting = value === nothing ? "$(pname) = $(base_val)  (unmodified baseline)" :
                                  "$(pname) = $(value)"
    @info @sprintf("[run %d/%d] %-24s | %s", _run_counter[], TOTAL_RUNS, tag, setting)
    cfg = AIHydroPoints.toml_read(base_toml)
    for f in cfg["data_settings"]["files"]              # make data paths absolute
        f["path"] = normpath(joinpath(base_dir, f["path"]))
    end
    value === nothing || _setpath!(cfg, param_path, value)
    _apply_output_overrides!(cfg; plot_outputs=plot_outputs_default, write_series=write_series_default)
    md = abspath(joinpath(sweep_dir, tag))
    cfg["model_settings"]["model_dir"] = md
    isdir(md) && for pf in readdir(md)                  # clear stale weights (no warm-start)
        occursin("params", pf) && rm(joinpath(md, pf))
    end
    tmp = joinpath(sweep_dir, "cfg_$tag.toml")
    AIHydroPoints.toml_write(tmp, cfg; overwrite=true)
    AIHydroPoints.train(tmp)
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
