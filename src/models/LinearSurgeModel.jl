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

`preprocess` returns a 1-tuple `(SurgeLagSource,)` — a lazy container over the
aligned forcing matrices.  Each training/predict minibatch is materialised as
`x_flat :: (3*nlocations_input*nlags, batchsize)` on demand, avoiding the
full-series feature matrix at 317 stations × 20 years.  The flux model is
`Chain(only, Dense(...))`: `only` unwraps the 1-tuple and `Dense` maps `x_flat`
to `(nlocations_output, batchsize)`.
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

Assemble the lazy Dense-input tuple from aligned forcing matrices.

Returns `((src::SurgeLagSource,), output)` where `src` holds raw stress/pressure
matrices and valid time indices; `materialize_batch` builds the flat feature
matrix per minibatch.

`forward` and `postprocess!` are inherited from `AbstractSurgeModel`; `predict`
and `train_model!` materialise batches automatically.
"""
function preprocess(model::LinearSurgeModel, input::Dict{String, TimeSeries})
    src, times_valid = _surge_lag_source(model, input)
    return (src,), _alloc_surge_output(model, times_valid)
end
