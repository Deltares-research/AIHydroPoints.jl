# ConvSurgeModel.jl
#
# Concrete subtype of AbstractSurgeModel using Conv1D layers over the lag
# dimension to predict storm surge from wind-stress and pressure history.
#
# Inherits from AbstractSurgeModel without override:
#   preprocess, postprocess!, train_model!, plot_series, save_params, load_params!

using Flux

"""
    ConvSurgeModel <: AbstractSurgeModel

Surge model that applies 1-D convolutions over the lag dimension of the
wind-stress and pressure history to predict storm surge.

## Constructor

```julia
model = ConvSurgeModel(settings::Dict{String, Any})
```

Required keys in `settings`: `"nstations"`, `"nwind"`, `"nlags"`.

Optional key `"model_pars"` (Dict):
- `"channels"` — output channels for each Conv1D layer (default `[32, 16]`)
- `"filtersize"` — Conv1D kernel width (default `3`)

## Architecture

```
(3*nwind*nlags, ntimes)          ← flattened lag tensor from preprocess
    reshape → (nlags, 3*nwind, ntimes)
    Conv1D(filtersize, 3*nwind  → channels[1], relu, SamePad)
    Conv1D(filtersize, channels[1] → channels[2], relu, SamePad)
    ...
    flatten → (nlags * channels[end], ntimes)
    Dense(nlags * channels[end] → nstations)
(nstations, ntimes)
```

Each Conv1D layer uses `pad=SamePad()` (stride 1) so the lag dimension is
preserved throughout, giving `Dense` a predictable input size of
`nlags * channels[end]`.
"""
mutable struct ConvSurgeModel <: AbstractSurgeModel
    flux_model
    settings :: Dict{String, Any}
end

"""
    ConvSurgeModel(settings::Dict{String, Any}) -> ConvSurgeModel

Construct a `ConvSurgeModel` from `settings`.

Required keys: `"nstations"` (Int), `"nwind"` (Int), `"nlags"` (Int).
"""
function ConvSurgeModel(settings::Dict{String, Any})
    nstations  = settings["nstations"]
    nwind      = settings["nwind"]
    nlags      = settings["nlags"]
    mp         = get(settings, "model_pars", Dict{String, Any}())
    channels   = get(mp, "channels",   [32, 16])
    filtersize = get(mp, "filtersize", 3)

    n_in    = 3 * nwind
    ch_seq  = [n_in; channels]

    chain = Chain(
        x -> reshape(x, nlags, n_in, size(x, 2)),
        [Conv((filtersize,), ch_seq[i] => ch_seq[i+1], relu; pad=SamePad())
         for i in 1:length(ch_seq)-1]...,
        Flux.flatten,
        Dense(nlags * channels[end] => nstations),
    )
    return ConvSurgeModel(chain, settings)
end

get_flux_model(m::ConvSurgeModel) = m.flux_model
get_settings(m::ConvSurgeModel)   = m.settings

"""
    forward(model::ConvSurgeModel, x::Array{Float32, 4}) -> Array{Float32, 3}

Flatten `x` to `(3*nwind*nlags, ntimes)`, run the Conv1D chain (which reshapes
internally), and return `(nstations, 1, ntimes)`.
"""
function forward(model::ConvSurgeModel, x::Array{Float32, 4})
    _, nfeatures, nlags_dim, ntimes = size(x)
    x_flat = reshape(x, nfeatures * nlags_dim, ntimes)
    y      = model.flux_model(x_flat)
    return reshape(y, size(y, 1), 1, ntimes)
end
