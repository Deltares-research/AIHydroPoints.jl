# DenseSurgeInteractionModel.jl
#
# Concrete AbstractSurgeInteractionModel.  An unstructured deep MLP over the
# concatenation of the flattened wind-stress/pressure forcing and the flattened
# tide lag windows:
#
#     y = MLP(vcat(f_flat, flatten(t_lags)))
#
# Unlike BiLinearSurgeInteractionModel's multiplicative gate (a narrow,
# physically-motivated interaction), this feeds tide and forcing into the same
# unstructured network with no built-in inductive bias about how they combine
# — the "+tide" variant of item 2's deep-MLP test in
# brainstorm_surge_model_status.md.

using Flux

# ──────────────────────────────────────────────────────────────────────────────
# DenseSurgeInteractionFlux — MLP over concatenated forcing + tide
# ──────────────────────────────────────────────────────────────────────────────

"""
    DenseSurgeInteractionFlux{C}

Flux model for `DenseSurgeInteractionModel`. Called on the tuple
`(f_flat, t_lags)`; flattens `t_lags :: (nlags, nstations, nbatch)` to
`(nlags*nstations, nbatch)`, concatenates it onto `f_flat` along the feature
axis, and runs the result through `chain` (a plain `Dense` MLP).
"""
struct DenseSurgeInteractionFlux{C}
    chain :: C
end

function (m::DenseSurgeInteractionFlux)(x)
    f_flat, t_lags = x
    nlags, nstations, nbatch = size(t_lags)
    t_flat = reshape(t_lags, nlags * nstations, nbatch)
    return m.chain(vcat(f_flat, t_flat))
end

Flux.@layer DenseSurgeInteractionFlux

# ──────────────────────────────────────────────────────────────────────────────
# DenseSurgeInteractionModel
# ──────────────────────────────────────────────────────────────────────────────

"""
    DenseSurgeInteractionModel <: AbstractSurgeInteractionModel

Dense (unstructured deep MLP) surge-interaction model: wind-stress/pressure
forcing and tide lag windows are flattened, concatenated into a single feature
vector, and passed through several `Dense` layers with a nonlinearity. No
built-in inductive bias about how tide and forcing combine — contrast with
`BiLinearSurgeInteractionModel`'s multiplicative per-station gate.

## Constructor

```julia
model = DenseSurgeInteractionModel(settings::Dict{String, Any})
```

Required keys: `"nlocations_output"`, `"nlocations_input"`, `"nlags"`.

Optional `model_pars`:

| Key | Default | Description |
|---|---|---|
| `"hidden_dims"` | `[64, 32]` | Sizes of the hidden layers. |
| `"activation"`  | `"relu"`   | Hidden-layer activation: `"relu"`, `"tanh"`, or `"swish"`. |
"""
mutable struct DenseSurgeInteractionModel <: AbstractSurgeInteractionModel
    flux_model :: DenseSurgeInteractionFlux
    settings   :: Dict{String, Any}
end

_dense_surge_interaction_activation(name::AbstractString) =
    name == "relu"  ? relu  :
    name == "tanh"  ? tanh  :
    name == "swish" ? swish :
    error("DenseSurgeInteractionModel: unknown activation \"$name\" " *
          "(use \"relu\", \"tanh\", or \"swish\").")

"""
    DenseSurgeInteractionModel(settings::Dict{String, Any}) -> DenseSurgeInteractionModel

Construct the model. Requires `"nlocations_output"`, `"nlocations_input"` and
`"nlags"` in `settings`; reads optional `model_pars` `"hidden_dims"` and `"activation"`.
"""
function DenseSurgeInteractionModel(settings::Dict{String, Any})
    nstations = settings["nlocations_output"]
    nwind     = settings["nlocations_input"]
    nlags     = settings["nlags"]

    mp          = get(settings, "model_pars", Dict{String, Any}())
    hidden_dims = get(mp, "hidden_dims", [64, 32])
    act         = _dense_surge_interaction_activation(get(mp, "activation", "relu"))

    n_in   = 3 * nwind * nlags + nlags * nstations   # flattened forcing + flattened tide
    dims   = [n_in; hidden_dims]
    layers = Any[Dense(dims[i] => dims[i+1], act) for i in 1:length(dims)-1]
    push!(layers, Dense(dims[end] => nstations))     # identity activation: allow negative surge

    flux = DenseSurgeInteractionFlux(Chain(layers...))
    return DenseSurgeInteractionModel(flux, settings)
end

get_flux_model(m::DenseSurgeInteractionModel) = m.flux_model
get_settings(m::DenseSurgeInteractionModel)   = m.settings

# preprocess (both forms), postprocess!, forward, train_model!, save/load are all
# inherited from AbstractSurgeInteractionModel / AbstractSurgeModel / AbstractFluxModel.
