# AbstractInteractionModel.jl
#
# Intermediate abstract type for all interaction models.  Sits between
# AbstractFluxModel and concrete models (ConvInteractionModel, etc.) and
# implements the shared interaction-specific logic.
#
# The interaction model predicts waterlevel from lagged tide and surge at each
# station.  Station identity is encoded via one-hot vectors so each
# (station × time) pair is an independent sample, exactly as in wave models.
#
# Input dict keys:  "tide", "surge"
# Target dict keys: "waterlevel"
# Output dict keys: "waterlevel"
#
# Normalization: Z-score, computed from training data and stored in settings
# under "input_mu", "input_std", "output_mu", "output_std".
#
# Concrete subtypes must implement: get_flux_model, get_settings.
# (forward, postprocess!, and train_model! are shared here.)

using Flux
using Printf: @sprintf
using ProgressMeter: Progress, next!
using Statistics: mean, std

# ──────────────────────────────────────────────────────────────────────────────
# Abstract type
# ──────────────────────────────────────────────────────────────────────────────

"""
    AbstractInteractionModel <: AbstractFluxModel

Abstract supertype for tide-surge interaction models.  Provides shared
implementations of `preprocess`, `postprocess!`, `train_model!`, and
`plot_series`.

## Required settings keys

| Key | Description |
|---|---|
| `"nlocations_output"` | Number of output (waterlevel) locations |
| `"nlags"`     | Number of lagged time steps used as input |

The following are populated automatically by `train_model!` on first call:
`"out_names"`, `"out_lons"`, `"out_lats"`, `"out_quantity"`, and the
Z-score normalisation statistics `"input_mu"`, `"input_std"`, `"output_mu"`,
`"output_std"`.

## Tensor layout

`preprocess` produces `(x_station, x_ts)` where:
- `x_station :: Bool (nstations, nstations * ntimes_valid)` — one-hot station encoding
- `x_ts :: Float32 (nlags, 2, nstations * ntimes_valid)` — Z-scored tide+surge lags,
  with channel 1 = surge and channel 2 = tide for each station.

Samples are ordered: for each valid time step (outer), then for each station (inner).
`forward` is provided generically (`get_flux_model(m)(x)`) and returns a 2-D
`(1, nstations*ntimes_valid)` array — one Z-scored value per `(station, time)`
sample; `postprocess!` reshapes to `(nstations, ntimes_valid)` and inverts the
Z-score.

## Input convention

`input` must contain:
- `"tide"`  — tidal signal `(nstations, ntimes)`
- `"surge"` — storm surge `(nstations, ntimes)`

`target` must contain:
- `"waterlevel"` — observed waterlevel `(nstations, ntimes)` (also carries station metadata)

## Concrete subtypes must implement

- `get_flux_model(m)` — return the underlying Flux model
- `get_settings(m)` — return `Dict{String, Any}`

`forward`, `postprocess!`, and `train_model!` are shared; the Flux model must
accept the `(x_station, x_ts)` tuple as a single argument and return a 2-D
`(1, nstations*ntimes_valid)` array.
"""
abstract type AbstractInteractionModel <: AbstractFluxModel end

# ──────────────────────────────────────────────────────────────────────────────
# Private helper: build raw (unnormalized) input blocks
# ──────────────────────────────────────────────────────────────────────────────

function _build_interaction_blocks(model::AbstractInteractionModel,
                                   input::Dict{String, TimeSeries})
    settings  = get_settings(model)
    nlags     = settings["nlags"]

    ts_tide  = input["tide"]
    ts_surge = input["surge"]

    tide_vals  = get_values(ts_tide)    # (nstations, ntimes)
    surge_vals = get_values(ts_surge)   # (nstations, ntimes)

    nstations    = size(tide_vals, 1)
    ntimes       = size(tide_vals, 2)
    ntimes_valid = ntimes - nlags + 1
    nsamples     = nstations * ntimes_valid

    # x_ts: (nlags, 2, nsamples)  — channel 1 = surge, channel 2 = tide
    x_ts = zeros(Float32, nlags, 2, nsamples)
    for (t_idx, itime) in enumerate(nlags:ntimes)
        for s in 1:nstations
            isample = (t_idx - 1) * nstations + s
            x_ts[:, 1, isample] .= surge_vals[s, itime-nlags+1:itime]
            x_ts[:, 2, isample] .= tide_vals[s,  itime-nlags+1:itime]
        end
    end

    # One-hot station encoding: (nstations, nstations * ntimes_valid)
    station_arr = collect(1:nstations) * ones(Int, ntimes_valid)'
    x_station   = Float32.(Flux.onehotbatch(station_arr[:], 1:nstations))

    return x_station, x_ts
end

# ──────────────────────────────────────────────────────────────────────────────
# preprocess — shared across all interaction models
# ──────────────────────────────────────────────────────────────────────────────

"""
    preprocess(model::AbstractInteractionModel, input::Dict{String, TimeSeries})
        -> (Tuple, Dict{String, TimeSeries})

Build one-hot station and lagged tide+surge tensors from `input`, Z-score
the input blocks using statistics stored in the model settings (default: no
scaling if statistics are not yet set), and pre-allocate the output
`TimeSeries`.

Returns `((x_station, x_ts), output)` where:
- `x_station :: Bool (nstations, nstations * ntimes_valid)` — one-hot
- `x_ts :: Float32 (nlags, 2, nstations * ntimes_valid)` — Z-scored tide+surge blocks
- `output` is `Dict("waterlevel" => ts)` with `ts.values` zero-initialised.

Station metadata is read from `settings["out_names"]` etc. when available,
falling back to the metadata of `input["tide"]`.
"""
function preprocess(model::AbstractInteractionModel, input::Dict{String, TimeSeries})
    settings  = get_settings(model)
    nlags     = settings["nlags"]
    input_mu  = Float32(get(settings, "input_mu",  0.0))
    input_std = Float32(get(settings, "input_std", 1.0))

    if haskey(settings, "out_names")
        out_names = settings["out_names"]
        input = Dict(k => _check_and_align_locations(v, out_names, "input[\"$k\"]")
                     for (k, v) in input)
    end

    x_station, x_ts = _build_interaction_blocks(model, input)
    x_ts_norm = (x_ts .- input_mu) ./ input_std

    nstations    = size(x_station, 1)
    times_valid  = get_times(input["tide"])[nlags:end]
    ntimes_valid = length(times_valid)

    ts_ref = input["tide"]
    names = get(settings, "out_names", get_names(ts_ref))
    lons  = get(settings, "out_lons",  Float64.(get_longitudes(ts_ref)))
    lats  = get(settings, "out_lats",  Float64.(get_latitudes(ts_ref)))
    qty   = get(settings, "out_quantity", "waterlevel")

    out_ts = TimeSeries(
        zeros(Float32, nstations, ntimes_valid),
        times_valid, names, lons, lats, qty, string(typeof(model)),
    )
    out_key = first(get(settings, "out_quantities", ["waterlevel"]))
    return (x_station, x_ts_norm), Dict{String, TimeSeries}(out_key => out_ts)
end

"""
    preprocess(model::AbstractInteractionModel, input, target) -> (Tuple, Matrix)

Train-form preprocess: build the input tuple `(x_station, x_ts_norm)` and the
Z-scored target `y :: (1, nstations*ntimes_valid)`.

On the **first** call (when the Z-score statistics are not yet in settings) the
statistics are fit from `(input, target)` and stored
(`input_mu`/`input_std`/`output_mu`/`output_std`); subsequent calls (the
validation set) reuse the stored values, so train and val share one normalisation
— matching the previous `train_model!` behaviour on the explicit-`val_input` path.

Consumed by the generic `train_model!`.
"""
function preprocess(model::AbstractInteractionModel, input::Dict{String, TimeSeries},
                    target::Dict{String, TimeSeries})
    settings = get_settings(model)
    nlags    = settings["nlags"]

    # Raw (unnormalised) input blocks + raw target (station × time samples).
    x_station, x_ts = _build_interaction_blocks(model, input)
    y_raw = reshape(Float32.(get_values(first(values(target))))[:, nlags:end], 1, :)

    # Fit Z-score stats on first call; reuse thereafter (so val inherits train stats).
    if !haskey(settings, "input_mu")
        settings["input_mu"]   = Float32(mean(x_ts))
        settings["input_std"]  = max(Float32(std(x_ts)),  1f-6)
        settings["output_mu"]  = Float32(mean(y_raw))
        settings["output_std"] = max(Float32(std(y_raw)), 1f-6)
    end
    input_mu   = Float32(settings["input_mu"])
    input_std  = Float32(settings["input_std"])
    output_mu  = Float32(settings["output_mu"])
    output_std = Float32(settings["output_std"])

    x_i = (x_ts  .- input_mu)  ./ input_std
    y   = (y_raw .- output_mu) ./ output_std
    return (x_station, x_i), y
end

# ──────────────────────────────────────────────────────────────────────────────
# postprocess! — shared across all interaction models
# (forward and train_model! are inherited from AbstractFluxModel)
# ──────────────────────────────────────────────────────────────────────────────

"""
    postprocess!(output::Dict{String, TimeSeries}, model::AbstractInteractionModel,
                 y::AbstractMatrix)

Write waterlevel predictions from the 2-D `y` of shape
`(1, nstations*ntimes_valid)` into the output in-place: reshape the
`(station × time)` samples back to `(nstations, ntimes_valid)` (station-fastest
within time, matching `preprocess`) and apply the inverse Z-score transform.
"""
function postprocess!(output::Dict{String, TimeSeries}, model::AbstractInteractionModel,
                      y::AbstractMatrix)
    settings   = get_settings(model)
    output_mu  = Float32(get(settings, "output_mu",  0.0))
    output_std = Float32(get(settings, "output_std", 1.0))
    out_key = first(keys(output))
    vals              = output[out_key].values          # (nstations, ntimes_valid)
    nstations, ntimes = size(vals)
    vals .= reshape(y, nstations, ntimes) .* output_std .+ output_mu
end

