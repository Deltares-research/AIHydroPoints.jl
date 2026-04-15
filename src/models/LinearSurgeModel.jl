# LinearSurgeModel.jl
#
# Concrete subtype of AbstractFluxModel for a linear surge model.
# Uses a single Flux.Dense layer (identity activation) to predict storm surge
# at nstations output locations from wind-stress and pressure forcing at nwind
# input locations over nlags previous time steps.
#
# Purpose: prototype / sanity-check for the AbstractFluxModel interface.
# It is intentionally minimal — no station encoding, no normalisation.

using Flux
using JLD2

# ──────────────────────────────────────────────────────────────────────────────
# Struct
# ──────────────────────────────────────────────────────────────────────────────

"""
    LinearSurgeModel <: AbstractFluxModel

A minimal linear surge model: a single `Dense` layer (identity activation) that
maps flattened wind-stress and pressure history to storm-surge predictions.

Input variables expected in `predict` / `preprocess`:
- `"wind_x"`     — east wind-stress component  (`TimeSeries`, shape `(nwind, T)`)
- `"wind_y"`     — north wind-stress component (`TimeSeries`, shape `(nwind, T)`)
- `"pressure"`   — sea-level pressure          (`TimeSeries`, shape `(nwind, T)`)
- `"waterlevel"` — used only for output station metadata (names, lon, lat)

## Constructor

```julia
model = LinearSurgeModel(settings::Dict{String, Any})
```

Required keys in `settings`: `"nstations"`, `"nwind"`, `"nlags"`.

## Tensor layout

`preprocess` returns `(tensor, output)` where:
- `tensor` has shape `(1, 3*nwind, nlags, ntimes_valid)`: one forcing "location",
  `3*nwind` channels (wind_x, wind_y, pressure), `nlags` lag steps.
- `output` is `Dict("surge" => ts)` with a zero-initialised `TimeSeries`.

`forward` flattens to `(3*nwind*nlags, ntimes_valid)`, applies `Dense`, and
reshapes to `(nstations, 1, ntimes_valid)`.

`postprocess!` writes `y[:, 1, :]` into `output["surge"].values` in-place.
"""
mutable struct LinearSurgeModel <: AbstractFluxModel
    flux_model
    settings :: Dict{String, Any}
end

"""
    LinearSurgeModel(settings::Dict{String, Any}) -> LinearSurgeModel

Construct an uninitialised `LinearSurgeModel` from `settings`.

Required keys: `"nstations"` (Int), `"nwind"` (Int), `"nlags"` (Int).
"""
function LinearSurgeModel(settings::Dict{String, Any})
    nstations = settings["nstations"]
    nwind     = settings["nwind"]
    nlags     = settings["nlags"]
    chain     = Dense(3 * nwind * nlags => nstations)   # linear (identity) output
    return LinearSurgeModel(chain, settings)
end

# ──────────────────────────────────────────────────────────────────────────────
# Required interface: get_flux_model, get_settings
# ──────────────────────────────────────────────────────────────────────────────

get_flux_model(m::LinearSurgeModel) = m.flux_model
get_settings(m::LinearSurgeModel)   = m.settings

# ──────────────────────────────────────────────────────────────────────────────
# preprocess
# ──────────────────────────────────────────────────────────────────────────────

"""
    preprocess(model::LinearSurgeModel, input::Dict{String, TimeSeries})
        -> (Array{Float32, 4}, Dict{String, TimeSeries})

Assemble a lag-window tensor from wind-stress and pressure forcing, and
pre-allocate the output `TimeSeries` for surge.

Returns `(tensor, output)` where:
- `tensor` has shape `(1, 3*nwind, nlags, ntimes_valid)`,
  with `ntimes_valid = length(times) - nlags + 1`.
- `output` is `Dict("surge" => ts)` with `ts.values` initialised to zeros.

Pressure is scaled by `2e-4*(p - 1e5)` to match the order of magnitude of the
wind-stress components (same convention as `surge.jl`).
"""
function preprocess(model::LinearSurgeModel, input::Dict{String, TimeSeries})
    nwind = model.settings["nwind"]
    nlags = model.settings["nlags"]

    wind_x = Float32.(get_values(input["wind_x"]))
    wind_y = Float32.(get_values(input["wind_y"]))
    press  = Float32.(2e-4 .* (get_values(input["pressure"]) .- 1e5))

    times       = get_times(input["wind_x"])
    ntimes      = length(times)
    valid_range = nlags:ntimes
    nvalid      = length(valid_range)

    # Assemble lag windows: (3*nwind, nlags, nvalid)
    x = zeros(Float32, 3 * nwind, nlags, nvalid)
    for (i, t) in enumerate(valid_range)
        x[1:nwind,           :, i] = wind_x[:, t-nlags+1:t]
        x[nwind+1:2*nwind,   :, i] = wind_y[:, t-nlags+1:t]
        x[2*nwind+1:3*nwind, :, i] = press[ :, t-nlags+1:t]
    end

    # Pre-allocate output TimeSeries with zero values
    wl        = input["waterlevel"]
    nstations = model.settings["nstations"]
    out_ts    = TimeSeries(
        zeros(Float32, nstations, nvalid),
        times[valid_range],
        get_names(wl),
        Float64.(get_longitudes(wl)),
        Float64.(get_latitudes(wl)),
        "surge",
        "LinearSurgeModel",
    )
    output = Dict{String, TimeSeries}("surge" => out_ts)

    # Add leading "locations" dimension (one forcing location)
    tensor = reshape(x, 1, 3 * nwind, nlags, nvalid)
    return tensor, output
end

# ──────────────────────────────────────────────────────────────────────────────
# forward
# ──────────────────────────────────────────────────────────────────────────────

"""
    forward(model::LinearSurgeModel, x::Array{Float32, 4})
        -> Array{Float32, 3}

Flatten the last three dimensions to `(3*nwind*nlags, ntimes)`, apply the
`Dense` layer, and reshape to `(nstations, 1, ntimes)`.
"""
function forward(model::LinearSurgeModel, x::Array{Float32, 4})
    _, nfeatures, nlags_dim, ntimes = size(x)
    x_flat = reshape(x, nfeatures * nlags_dim, ntimes)  # (3*nwind*nlags, ntimes)
    y      = model.flux_model(x_flat)                    # (nstations, ntimes)
    return reshape(y, size(y, 1), 1, ntimes)             # (nstations, 1, ntimes)
end

# ──────────────────────────────────────────────────────────────────────────────
# postprocess!
# ──────────────────────────────────────────────────────────────────────────────

"""
    postprocess!(output::Dict{String, TimeSeries}, model::LinearSurgeModel,
                 y::Array{Float32, 3})

Write the surge predictions from `y` into the pre-allocated `output["surge"]`
in-place.  `y` has shape `(nstations, 1, ntimes)`; the singleton feature
dimension is dropped before writing.
"""
function postprocess!(output::Dict{String, TimeSeries}, model::LinearSurgeModel,
                      y::Array{Float32, 3})
    output["surge"].values .= y[:, 1, :]
end
