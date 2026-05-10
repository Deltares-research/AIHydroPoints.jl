# LinearSurgeModel.jl
#
# Concrete subtype of AbstractSurgeModel for a linear surge model.
# Uses a single Flux.Dense layer (identity activation) to predict storm surge
# at nlocations_output locations from wind-stress and pressure forcing at
# nlocations_input locations over nlags previous time steps.

using Flux

"""
    LinearSurgeModel <: AbstractSurgeModel

A minimal linear surge model: a single `Dense` layer (identity activation) that
maps flattened wind-stress and pressure history to storm-surge predictions.

## Constructor

```julia
model = LinearSurgeModel(settings::Dict{String, Any})
```

Required keys in `settings`: `"nlocations_output"`, `"nlocations_input"`, `"nlags"`.

## Tensor layout

`forward` flattens the `(1, 3*nlocations_input, nlags, ntimes_valid)` tensor from
`preprocess` to `(3*nlocations_input*nlags, ntimes_valid)`, applies `Dense`, and reshapes
to `(nlocations_output, 1, ntimes_valid)`.
"""
mutable struct LinearSurgeModel <: AbstractSurgeModel
    flux_model
    settings :: Dict{String, Any}
end

"""
    LinearSurgeModel(settings::Dict{String, Any}) -> LinearSurgeModel

Construct a `LinearSurgeModel` from `settings`.

Required keys: `"nlocations_output"` (Int), `"nlocations_input"` (Int), `"nlags"` (Int).
"""
function LinearSurgeModel(settings::Dict{String, Any})
    nstations = settings["nlocations_output"]
    nwind     = settings["nlocations_input"]
    nlags     = settings["nlags"]
    chain     = Dense(3 * nwind * nlags => nstations)
    return LinearSurgeModel(chain, settings)
end

get_flux_model(m::LinearSurgeModel) = m.flux_model
get_settings(m::LinearSurgeModel)   = m.settings

"""
    forward(model::LinearSurgeModel, x::Array{Float32, 4}) -> Array{Float32, 3}

Flatten `x` to `(3*nwind*nlags, ntimes)`, apply the `Dense` layer, and reshape
to `(nstations, 1, ntimes)`.
"""
function forward(model::LinearSurgeModel, x::Array{Float32, 4})
    _, nfeatures, nlags_dim, ntimes = size(x)
    x_flat = reshape(x, nfeatures * nlags_dim, ntimes)
    y      = model.flux_model(x_flat)
    return reshape(y, size(y, 1), 1, ntimes)
end
