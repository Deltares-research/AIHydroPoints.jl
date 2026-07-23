# AttentionSurgeModel.jl
#
# Concrete subtype of AbstractSurgeModel using the attention-based surge
# architecture (branch/trunk/downsample with graph adjacency).
#
# Inherits from AbstractSurgeModel:
#   forward, postprocess!, train_model!, save_params, load_params!, predict
#
# Overrides only preprocess (two input tensors instead of one). The generic
# forward/train_model! handle the 2-input case because preprocess returns a
# tuple and the Flux model is called by splatting it (m(x_station, x_wind)).

using Flux
using Dates

# ──────────────────────────────────────────────────────────────────────────────
# Internal Flux architecture
# ──────────────────────────────────────────────────────────────────────────────

"""
    AttentionSurgeFlux

Internal Flux model for `AttentionSurgeModel`. Combines a transformer branch
network (processing wind/pressure history) with a trunk network (processing
station metadata) via a graph-adjacency-weighted merge.

Not exported — construct via `AttentionSurgeModel`.
"""
struct AttentionSurgeFlux{P, Q, R, T}
    branch_net :: P
    trunk_net  :: Q
    downsample :: R
    adjacency  :: AbstractArray{T, 2}
end

function (m::AttentionSurgeFlux)(x_station, x_wind)
    nbatch     = size(x_station)[end]
    branch_out = m.branch_net(x_wind)
    trunk_out  = m.trunk_net(x_station)
    nwind      = size(branch_out, 1)
    merged = batched_mul(
        batched_transpose(trunk_out .* m.adjacency),
        reshape(branch_out, (nwind, :, nbatch)),
    )
    # downsample → (nstations, nlags, nbatch); the prediction is the final lag,
    # so slice it here and return the 2-D (nstations, nbatch) surge output that
    # the shared surge pipeline (forward / train_model!) expects.
    return m.downsample(merged)[:, end, :]
end

# Tuple-call form so the generic train_model!/forward can invoke every flux model
# uniformly as `m(x)`.
(m::AttentionSurgeFlux)(x::Tuple) = m(x...)

@Flux.layer AttentionSurgeFlux

# ──────────────────────────────────────────────────────────────────────────────
# AttentionSurgeModel struct and constructor
# ──────────────────────────────────────────────────────────────────────────────

"""
    AttentionSurgeModel <: AbstractSurgeModel

Surge model using a transformer branch network for wind/pressure history and a
dense trunk network for station metadata, merged via graph adjacency weights.

## Constructor

```julia
model = AttentionSurgeModel(settings::Dict{String, Any}, gn::GraphNetwork)
```

Required keys in `settings`: `"nlocations_output"`, `"nlocations_input"`, `"nlags"`, `"model_pars"`.

Required keys in `"model_pars"`:
- `"nembed"`, `"theta"`, `"nheads"`, `"nlayers_branch"`, `"nlayers_trunk"`, `"nhidden_trunk"`

## Input variables

Same as `AbstractSurgeModel`: `"wind_x"`, `"wind_y"`, `"pressure"`.
Station encoding (lat/lon, time) is built automatically in `preprocess` from
`model.settings` and the input times.
"""
mutable struct AttentionSurgeModel <: AbstractSurgeModel
    flux_model :: AttentionSurgeFlux
    settings   :: Dict{String, Any}
end

"""
    AttentionSurgeModel(settings::Dict{String, Any}, gn::GraphNetwork)
        -> AttentionSurgeModel

Construct an `AttentionSurgeModel` from `settings` and a `GraphNetwork` for
the spatial adjacency structure.
"""
function AttentionSurgeModel(settings::Dict{String, Any}, gn::GraphNetwork)
    nlags   = settings["nlags"]
    nwind   = settings["nlocations_input"]
    mp      = settings["model_pars"]
    nembed          = mp["nembed"]
    theta           = mp["theta"]
    nheads          = mp["nheads"]
    nlayers_branch  = mp["nlayers_branch"]
    nlayers_trunk   = mp["nlayers_trunk"]
    nhidden_trunk   = mp["nhidden_trunk"]

    embed     = Embedder(3 * nwind, nembed)
    deembed   = Deembedder(embed)
    pos_embed = SinCosPosEmbedder(nembed, nlags; theta)

    branch_net = Chain(
        embed,
        pos_embed,
        [Transformer(nembed, nheads) for _ in 1:nlayers_branch]...,
        deembed,
        x -> reshape(x, (nwind, 3, nlags, :)),
    )

    trunk_net = Chain(
        Dense(6 => nhidden_trunk),
        [Dense(nhidden_trunk => nhidden_trunk) for _ in 1:nlayers_trunk]...,
        Dense(nhidden_trunk => nwind),
    )

    downsample = Conv((1,), 3 * nlags => nlags, identity; stride=(1,), pad=SamePad())

    flux_model = AttentionSurgeFlux(branch_net, trunk_net, downsample, gn.adjacency)
    return AttentionSurgeModel(flux_model, settings)
end

get_flux_model(m::AttentionSurgeModel) = m.flux_model
get_settings(m::AttentionSurgeModel)   = m.settings

# ──────────────────────────────────────────────────────────────────────────────
# preprocess — overrides AbstractSurgeModel (two input tensors, not one)
# ──────────────────────────────────────────────────────────────────────────────

"""
    preprocess(model::AttentionSurgeModel, input::Dict{String, TimeSeries})
        -> (Tuple, Dict{String, TimeSeries})

Build the station-encoding tensor and the wind/pressure lag tensor, and
pre-allocate the surge output.

Returns `((x_station, x_wind), output)` where:
- `x_station :: (6, nstations, ntimes_valid)` — cos/sin of latitude, longitude,
  and day-of-year for each output station at each valid batch-time step.
- `x_wind :: (3*nlocations_input, nlags, ntimes_valid)` — the shared lag windows
  stacked along the feature axis in `(stress_x, stress_y, pressure)` order, so
  point varies fastest within quantity. The branch network's downstream
  `reshape(·, (nwind, 3, nlags, :))` relies on exactly this ordering.

Batch-time is the last axis of both tensors, so the shared `train_model!` can
batch them together. `forward` and `postprocess!` are inherited from
`AbstractSurgeModel`.

Requires `"out_lats"`, `"out_lons"` in `model.settings` (set automatically by
`train_model!` on first call).
"""
function preprocess(model::AttentionSurgeModel, input::Dict{String, TimeSeries})
    settings  = get_settings(model)
    nstations = settings["nlocations_output"]

    # Shared extraction: sx, sy, pr :: (nwind, nlags, nvalid), point-fastest.
    sx, sy, pr, times_valid = _surge_lag_windows(model, input)
    nvalid = length(times_valid)

    # x_wind: (3*nwind, nlags, nvalid) — point-fastest within quantity.
    x_wind = vcat(sx, sy, pr)

    # Station encoding: (6, nstations, nvalid) — cos/sin of lat, lon, day-of-year.
    lats      = settings["out_lats"]
    lons      = settings["out_lons"]
    dayperiod = 365.25
    times_day = Dates.dayofyear.(times_valid)
    times_cos = Float32.(cos.(2π .* times_day ./ dayperiod))'   # (1, nvalid)
    times_sin = Float32.(sin.(2π .* times_day ./ dayperiod))'

    x_station = zeros(Float32, 6, nstations, nvalid)
    x_station[1, :, :] .= Float32.(cos.(deg2rad.(lats)))
    x_station[2, :, :] .= Float32.(sin.(deg2rad.(lats)))
    x_station[3, :, :] .= Float32.(cos.(deg2rad.(lons)))
    x_station[4, :, :] .= Float32.(sin.(deg2rad.(lons)))
    x_station[5, :, :] .= times_cos
    x_station[6, :, :] .= times_sin

    return (x_station, x_wind), _alloc_surge_output(model, times_valid)
end
