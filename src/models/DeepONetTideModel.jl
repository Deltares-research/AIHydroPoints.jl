# DeepONetTideModel.jl
#
# Concrete subtype of AbstractTideModel wrapping the TideModel Flux architecture
# (branch/trunk/downsample DeepONet) from src/tides.jl.
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

"""
    forward(model::DeepONetTideModel, x::Tuple) -> Array{Float32, 3}

Unpack `(x_station, x_doodson)` from `x`, run `TideModel`, and return
predictions reshaped to `(nstations, 1, ntimes)`.
"""
function forward(model::DeepONetTideModel, x::Tuple)
    x_station, x_doodson = x
    y = model.flux_model(x_station, x_doodson)   # (nstations, ntimes)
    return reshape(y, size(y, 1), 1, size(y, 2))  # (nstations, 1, ntimes)
end
