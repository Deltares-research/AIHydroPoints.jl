# AbstractTideModel.jl
#
# Intermediate abstract type for all tide models.  Sits between AbstractFluxModel
# and concrete tide models (DeepONetTideModel, etc.) and implements shared
# tide-specific logic: preprocess, postprocess!, and train_model!.
#
# Concrete subtypes must implement: get_flux_model, get_settings.
# (forward, postprocess!, and train_model! are shared here.)
#
# Unlike surge models, tide models have no external forcing — inputs are computed
# from station lat/lon and time (Doodson numbers).  The input dict and target dict
# both carry "waterlevel" as the sole key.

using Flux
using Printf: @sprintf
using ProgressMeter: Progress, next!

"""
    AbstractTideModel <: AbstractFluxModel

Abstract supertype for tide models.  Provides shared implementations of
`preprocess`, `postprocess!`, and `train_model!` for models that predict
tides from station coordinates and astronomical time (Doodson numbers).

## Required settings keys

| Key | Description |
|---|---|
| `"freqs"` | Vector of tidal constituent names, e.g. `["M2","S2","K1",...]` |
| `"model_pars"` | Dict of architecture hyperparameters (subtype-specific) |

The following are populated automatically by `train_model!` on first call:
`"out_names"`, `"out_lons"`, `"out_lats"`, `"out_quantity"`.

## Tensor layout

`preprocess` produces `(x_station, x_doodson)` where:
- `x_station :: Float32 (4, nstations, ntimes)` — cos/sin lat and cos/sin lon per station
- `x_doodson :: Float32 (2*nfreqs, ntimes)` — cos/sin Doodson arguments per time step

`forward` is provided generically here (`get_flux_model(m)(x...)`); the Flux
model must return a 2-D `(nstations, ntimes)` array.

## Input convention

Both `input` and `target` dicts carry a `"waterlevel"` key.
`preprocess` reads times and station metadata from `input["waterlevel"]`.
`train_model!` reads ground truth values from `target["waterlevel"]`.

## Concrete subtypes must implement

- `get_flux_model(m)` — return the underlying Flux model
- `get_settings(m)` — return `Dict{String, Any}`

`forward`, `postprocess!`, and `train_model!` are shared; the Flux model must
return a 2-D `(nstations, ntimes)` array.
"""
abstract type AbstractTideModel <: AbstractFluxModel end

# ──────────────────────────────────────────────────────────────────────────────
# preprocess — shared across all tide models
# ──────────────────────────────────────────────────────────────────────────────

"""
    preprocess(model::AbstractTideModel, input::Dict{String, TimeSeries})
        -> (Tuple, Dict{String, TimeSeries})

Build station-encoding and Doodson-argument tensors from `input["waterlevel"]`,
and pre-allocate the output `TimeSeries`.

Returns `((x_station, x_doodson), output)` where:
- `x_station :: Float32 (4, nstations, ntimes)` — `[cos(lat), sin(lat), cos(lon), sin(lon)]`
- `x_doodson :: Float32 (2*nfreqs, ntimes)` — `[cos(doodson); sin(doodson)]`
- `output` is `Dict("waterlevel" => ts)` with `ts.values` zero-initialised.

Tidal frequencies are taken from `model.settings["freqs"]`.
"""
function preprocess(model::AbstractTideModel, input::Dict{String, TimeSeries})
    settings  = get_settings(model)
    ts        = input["waterlevel"]
    times     = get_times(ts)
    lats      = get_latitudes(ts)
    lons      = get_longitudes(ts)
    names     = get_names(ts)
    nstations = length(lats)
    ntimes    = length(times)
    freqs     = settings["freqs"]
    nfreqs    = length(freqs)

    x_station = zeros(Float32, 4, nstations, ntimes)
    x_station[1, :, :] .= Float32.(cos.(deg2rad.(lats)))
    x_station[2, :, :] .= Float32.(sin.(deg2rad.(lats)))
    x_station[3, :, :] .= Float32.(cos.(deg2rad.(lons)))
    x_station[4, :, :] .= Float32.(sin.(deg2rad.(lons)))

    frequencies = primary_frequencies_as_doodson(freqs)
    doodson     = (get_doodson_eqvals(times) * frequencies)'   # (nfreqs, ntimes)
    x_doodson   = Float32.(vcat(cos.(doodson), sin.(doodson))) # (2*nfreqs, ntimes)

    out_ts = TimeSeries(
        zeros(Float32, nstations, ntimes),
        times,
        names,
        Float64.(lons),
        Float64.(lats),
        get_quantity(ts),
        string(typeof(model)),
    )
    output = Dict{String, TimeSeries}("waterlevel" => out_ts)

    return (x_station, x_doodson), output
end

"""
    preprocess(model::AbstractTideModel, input, target) -> (Tuple, Matrix)

Train-form preprocess: build the input tuple `(x_station, x_doodson)` (reusing
the 2-arg predict form) and the target `y = get_values(target["waterlevel"])`.

Tide models predict per time step for **all** times, so there is no lag window
and no normalisation — `y` is the full `(nstations, ntimes)` series.  Consumed by
the generic `train_model!`.
"""
function preprocess(model::AbstractTideModel, input::Dict{String, TimeSeries},
                    target::Dict{String, TimeSeries})
    x, _ = preprocess(model, input)   # reuse 2-arg for the input tuple
    y = Float32.(get_values(first(values(target))))   # (nstations, ntimes)
    return x, y
end

# ──────────────────────────────────────────────────────────────────────────────
# postprocess! — shared across all tide models
# (forward and train_model! are inherited from AbstractFluxModel)
# ──────────────────────────────────────────────────────────────────────────────

"""
    postprocess!(output::Dict{String, TimeSeries}, model::AbstractTideModel,
                 y::AbstractMatrix)

Write the 2-D tide predictions `y` of shape `(nstations, ntimes)` into
`output["waterlevel"]` in-place.
"""
function postprocess!(output::Dict{String, TimeSeries}, model::AbstractTideModel,
                      y::AbstractMatrix)
    output["waterlevel"].values .= y
end
