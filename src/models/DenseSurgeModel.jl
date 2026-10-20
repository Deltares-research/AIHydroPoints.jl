# DenseSurgeModel.jl
#
# Concrete subtype of AbstractSurgeModel: an unstructured deep MLP (multiple
# Dense layers with a nonlinearity) mapping flattened wind-stress and pressure
# history to storm-surge predictions. Same input/output convention as
# LinearSurgeModel; LinearSurgeModel is the special case hidden_dims = [].

using Flux

"""
    DenseSurgeModel <: AbstractSurgeModel

An unstructured deep MLP surge model: several `Dense` layers with a
nonlinearity between them, mapping flattened wind-stress and pressure history
to storm-surge predictions. Generalises `LinearSurgeModel` (a single `Dense`
layer) to test whether depth/nonlinearity helps beyond a linear map, without
adding tide or any other input (see `brainstorm_surge_model_status.md`, item 2).

## Constructor

```julia
model = DenseSurgeModel(settings::Dict{String, Any})
```

Required keys in `settings`: `"nlocations_output"`, `"nlocations_input"`, `"nlags"`.

Optional key `"model_pars"` (Dict):
- `"hidden_dims"` — sizes of the hidden layers (default `[64, 32]`); an empty
  vector reduces the model to `LinearSurgeModel`.
- `"activation"` — hidden-layer activation, `"relu"` (default), `"tanh"`, or `"swish"`.

## Tensor layout

`preprocess` returns a 1-tuple `(x_flat,)` with
`x_flat :: (3*nlocations_input*nlags, ntimes_valid)`, identical to
`LinearSurgeModel`. The flux model is
`Chain(only, Dense(... , act), ..., Dense(... => nlocations_output))`: `only`
unwraps the 1-tuple, the hidden `Dense` layers apply `act`, and the final
`Dense` layer uses identity activation so surge predictions can be negative.
"""
mutable struct DenseSurgeModel <: AbstractSurgeModel
    flux_model
    settings :: Dict{String, Any}
end

_dense_surge_activation(name::AbstractString) =
    name == "relu"  ? relu  :
    name == "tanh"  ? tanh  :
    name == "swish" ? swish :
    error("DenseSurgeModel: unknown activation \"$name\" (use \"relu\", \"tanh\", or \"swish\").")

"""
    DenseSurgeModel(settings::Dict{String, Any}) -> DenseSurgeModel

Construct a `DenseSurgeModel` from `settings`.

Required keys: `"nlocations_output"` (Int), `"nlocations_input"` (Int), `"nlags"` (Int).
"""
function DenseSurgeModel(settings::Dict{String, Any})
    nstations = settings["nlocations_output"]
    nwind     = settings["nlocations_input"]
    nlags     = settings["nlags"]

    mp          = get(settings, "model_pars", Dict{String, Any}())
    hidden_dims = get(mp, "hidden_dims", [64, 32])
    act         = _dense_surge_activation(get(mp, "activation", "relu"))

    n_in   = 3 * nwind * nlags
    dims   = [n_in; hidden_dims]
    layers = Any[Dense(dims[i] => dims[i+1], act) for i in 1:length(dims)-1]
    push!(layers, Dense(dims[end] => nstations))   # identity activation: allow negative surge

    # `only` unwraps the 1-tuple `(x_flat,)` from preprocess so the flux model is
    # callable uniformly as `chain(x)` (matching LinearSurgeModel).
    chain = Chain(only, layers...)
    return DenseSurgeModel(chain, settings)
end

get_flux_model(m::DenseSurgeModel) = m.flux_model
get_settings(m::DenseSurgeModel)   = m.settings

"""
    preprocess(model::DenseSurgeModel, input::Dict{String, TimeSeries})
        -> (Tuple, Dict{String, TimeSeries})

Assemble the flat MLP-input tuple from the shared lag windows. Identical to
`LinearSurgeModel`'s `preprocess` — `DenseSurgeModel` differs only in the depth
of the Flux chain, not the input layout.

`forward` and `postprocess!` are inherited from `AbstractSurgeModel`.
"""
function preprocess(model::DenseSurgeModel, input::Dict{String, TimeSeries})
    sx, sy, pr, times_valid = _surge_lag_windows(model, input)
    nvalid    = size(sx, 3)
    x_stacked = vcat(sx, sy, pr)                 # (3*nwind, nlags, nvalid)
    x_flat    = reshape(x_stacked, :, nvalid)    # (3*nwind*nlags, nvalid)
    return (x_flat,), _alloc_surge_output(model, times_valid)
end
