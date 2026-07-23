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

`preprocess` returns a 1-tuple `(x_flat,)` with
`x_flat :: (3*nlocations_input*nlags, ntimes_valid)` — a flat feature vector per
batch-time step. The flux model is `Chain(only, Dense(...))`: `only` unwraps the
1-tuple and `Dense` maps `x_flat` to `(nlocations_output, ntimes_valid)`.
`Dense` is permutation-invariant on its input vector, so the exact interleaving of
point/quantity/lag inside the flat vector is irrelevant as long as it is the same
at train and predict time (it is — both go through this `preprocess`).
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
    # `only` unwraps the 1-tuple `(x_flat,)` from preprocess so the flux model is
    # callable uniformly as `chain(x)` (matching the multi-input models).
    chain     = Chain(only, Dense(3 * nwind * nlags => nstations))
    return LinearSurgeModel(chain, settings)
end

get_flux_model(m::LinearSurgeModel) = m.flux_model
get_settings(m::LinearSurgeModel)   = m.settings

"""
    preprocess(model::LinearSurgeModel, input::Dict{String, TimeSeries})
        -> (Tuple, Dict{String, TimeSeries})

Assemble the flat Dense-input tuple from the shared lag windows.

Returns `((x_flat,), output)` where `x_flat` has shape
`(3*nlocations_input*nlags, ntimes_valid)`: the `(point, lag, batch-time)` stress
and pressure windows stacked along features and flattened to one vector per
batch-time step (batch-time is the last axis).

`forward` and `postprocess!` are inherited from `AbstractSurgeModel`.
"""
function preprocess(model::LinearSurgeModel, input::Dict{String, TimeSeries})
    # sx, sy, pr :: (nwind, nlags, nvalid)   — shared extraction
    sx, sy, pr, times_valid = _surge_lag_windows(model, input)
    nvalid    = size(sx, 3)
    x_stacked = vcat(sx, sy, pr)                 # (3*nwind, nlags, nvalid)
    x_flat    = reshape(x_stacked, :, nvalid)    # (3*nwind*nlags, nvalid)
    return (x_flat,), _alloc_surge_output(model, times_valid)
end
