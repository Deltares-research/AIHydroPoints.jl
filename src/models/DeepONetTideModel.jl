# DeepONetTideModel.jl
#
# Concrete subtype of AbstractTideModel wrapping the TideModel Flux architecture
# (branch/trunk/downsample DeepONet).
#
# Inherits from AbstractTideModel:
#   preprocess, postprocess!, train_model!
#
# Inherits from AbstractFluxModel:
#   predict, save_params, load_params!
#
# Implements:
#   get_flux_model, get_settings, forward

using Flux

# ──────────────────────────────────────────────────────────────────────────────
# TideModel — DeepONet branch/trunk/downsample Flux architecture
# ──────────────────────────────────────────────────────────────────────────────

struct TideModel{P, Q, T}
    branch     :: P
    trunk      :: Q
    downsample :: T
end

function TideModel(nfreqs, nlayers_branch, nhidden_branch,
                   nlayers_trunk, nhidden_trunk, nlayers_down, activ_func)
    branch = Chain(
        Dense(2*nfreqs, nhidden_branch, activ_func),
        [Dense(nhidden_branch, nhidden_branch, activ_func) for _ in 1:nlayers_branch]...,
        Dense(nhidden_branch, nhidden_branch, tanh),
    )
    trunk = Chain(
        Dense(4, nhidden_trunk, activ_func),
        [Dense(nhidden_trunk, nhidden_trunk, activ_func) for _ in 1:nlayers_trunk]...,
        Dense(nhidden_trunk, 2, tanh),
    )
    down = Chain(
        [Dense(nhidden_branch, nhidden_branch, activ_func) for _ in 1:nlayers_down]...,
        Dense(nhidden_branch, 1),
    )
    return TideModel(branch, trunk, down)
end

function (m::TideModel)(x_stations, x_doodson)
    branch_out = m.branch(x_doodson)
    trunk_out  = m.trunk(x_stations)
    merged = cat(
        [slice[1,:]' .* branch_out .+ slice[2,:]'
         for slice in eachslice(trunk_out, dims=2)]...,
        dims=3,
    )
    merged = permutedims(merged, (1, 3, 2))
    merged = m.downsample(merged)
    return Flux.flatten(merged)
end

@Flux.layer TideModel

# ──────────────────────────────────────────────────────────────────────────────
# DeepONetTideModel
# ──────────────────────────────────────────────────────────────────────────────

"""
    DeepONetTideModel <: AbstractTideModel

Tide model using a DeepONet-style branch/trunk/downsample architecture.
The branch network processes Doodson arguments (astronomical tidal forcing);
the trunk network processes station coordinates (lat/lon); they are merged
per station and downsampled to a scalar prediction.

## Constructor

```julia
model = DeepONetTideModel(settings::Dict{String, Any})
```

Required keys in `settings`:
- `"freqs"`: Vector of tidal constituent names, e.g. `["M2","S2","K1",...]`
- `"model_pars"`: Dict with keys `"nlayers_branch"`, `"nhidden_branch"`,
  `"nlayers_trunk"`, `"nhidden_trunk"`, `"nlayers_down"`

## Input convention

Both `input` and `target` dicts carry a `"waterlevel"` key:

```julia
ts = NetCDFTimeSeries(...)
train_model!(model, train_settings,
             Dict("waterlevel" => ts),
             Dict("waterlevel" => ts))

output = predict(model, Dict("waterlevel" => ts_test))
```
"""
mutable struct DeepONetTideModel <: AbstractTideModel
    flux_model :: TideModel
    settings   :: Dict{String, Any}
end

"""
    DeepONetTideModel(settings::Dict{String, Any}) -> DeepONetTideModel

Construct a `DeepONetTideModel` from `settings`.

Required keys: `"freqs"`, `"model_pars"` (with `"nlayers_branch"`,
`"nhidden_branch"`, `"nlayers_trunk"`, `"nhidden_trunk"`, `"nlayers_down"`).
"""
function DeepONetTideModel(settings::Dict{String, Any})
    freqs  = settings["freqs"]
    nfreqs = length(freqs)
    mp     = settings["model_pars"]

    flux_model = TideModel(
        nfreqs,
        mp["nlayers_branch"],
        mp["nhidden_branch"],
        mp["nlayers_trunk"],
        mp["nhidden_trunk"],
        mp["nlayers_down"],
        leakyrelu,
    )
    return DeepONetTideModel(flux_model, settings)
end

get_flux_model(m::DeepONetTideModel) = m.flux_model
get_settings(m::DeepONetTideModel)   = m.settings

# forward, postprocess!, and train_model! are inherited from AbstractTideModel.
