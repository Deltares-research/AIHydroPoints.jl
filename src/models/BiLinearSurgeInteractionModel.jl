# BiLinearSurgeInteractionModel.jl
#
# Concrete AbstractSurgeInteractionModel.  A linear surge model whose prediction is
# multiplied by a tide-driven, per-station modulation:
#
#     y_P = (W ⋆ x + b)_P · (1 + a · σ( Σ_t' V[P,t'] · z[t',P] ))
#
# with forcing x (all input stations → all output stations, all-to-all) and tide z
# at the output stations (one-to-one). See docs/background.md (Surge-interaction
# models). Zero-init V ⇒ modulation = 1 ⇒ the model starts as LinearSurgeModel.

using Flux

# ──────────────────────────────────────────────────────────────────────────────
# StationTideModulation — per-output-station lag weighting of the tide
# ──────────────────────────────────────────────────────────────────────────────

"""
    StationTideModulation{M, F}

Per-output-station tide modulation.  Holds a **bias-free** weight matrix
`V :: (nstations, nlags)` (one lag-weight vector per station, location-dependent),
a fixed branch scale `a`, and an activation `σ`.  Maps the tide lag windows
`t_lags :: (nlags, nstations, nbatch)` to `mod :: (nstations, nbatch)`:

```
mod[P, :] = 1 + a * σ.( Σ_t' V[P, t'] * t_lags[t', P, :] )
```

Only `V` is trainable (`a` is a fixed scalar, `σ` a function).
"""
struct StationTideModulation{M, F}
    V :: M          # (nstations, nlags)
    a :: Float32
    σ :: F
end

function (l::StationTideModulation)(t_lags)
    nlags, nstations, _ = size(t_lags)
    Vp  = reshape(permutedims(l.V, (2, 1)), nlags, nstations, 1)   # (nlags, nstations, 1)
    raw = dropdims(sum(Vp .* t_lags; dims = 1); dims = 1)          # (nstations, nbatch)
    return 1f0 .+ l.a .* l.σ.(raw)
end

Flux.@layer StationTideModulation

# ──────────────────────────────────────────────────────────────────────────────
# BiLinearSurgeInteractionFlux — linear surge × tide modulation
# ──────────────────────────────────────────────────────────────────────────────

"""
    BiLinearSurgeInteractionFlux{S, M}

Flux model for `BiLinearSurgeInteractionModel`: a linear surge branch
`surge :: Dense(3·nwind·nlags => nstations)` and a `StationTideModulation` branch.
Called on the tuple `(f_flat, t_lags)`; returns `surge_lin .* mod`.
"""
struct BiLinearSurgeInteractionFlux{S, M}
    surge :: S      # Dense(3*nwind*nlags => nstations)
    mod   :: M      # StationTideModulation
end

function (m::BiLinearSurgeInteractionFlux)(x)
    f_flat, t_lags = x
    surge_lin = m.surge(f_flat)          # (nstations, nbatch)  [all-to-all]
    return surge_lin .* m.mod(t_lags)    # ⊙ per-station modulation [one-to-one]
end

Flux.@layer BiLinearSurgeInteractionFlux

# ──────────────────────────────────────────────────────────────────────────────
# BiLinearSurgeInteractionModel
# ──────────────────────────────────────────────────────────────────────────────

"""
    BiLinearSurgeInteractionModel <: AbstractSurgeInteractionModel

Bilinear surge-interaction model: linear surge from wind-stress/pressure forcing,
multiplied by a per-station tide modulation `1 + a·σ(V ⋆ tide)`.

## Constructor

```julia
model = BiLinearSurgeInteractionModel(settings::Dict{String, Any})
```

Required keys: `"nlocations_output"`, `"nlocations_input"`, `"nlags"`.

Optional `model_pars`:

| Key | Default | Description |
|---|---|---|
| `"a"`              | `0.1`        | Fixed interaction-branch scale (not learnable). |
| `"mod_activation"` | `"identity"` | Modulation activation: `"identity"` (bilinear) or `"tanh"` (bounded). |

The modulation weights `V` are zero-initialised, so the model starts as an exact
`LinearSurgeModel` (modulation ≡ 1) and learns the tidal coupling from there.
"""
mutable struct BiLinearSurgeInteractionModel <: AbstractSurgeInteractionModel
    flux_model :: BiLinearSurgeInteractionFlux
    settings   :: Dict{String, Any}
end

# Resolve the modulation activation name to a function.
_mod_activation(name::AbstractString) =
    name == "tanh"     ? tanh :
    name == "identity" ? identity :
    error("BiLinearSurgeInteractionModel: unknown mod_activation \"$name\" " *
          "(use \"identity\" or \"tanh\").")

"""
    BiLinearSurgeInteractionModel(settings::Dict{String, Any}) -> BiLinearSurgeInteractionModel

Construct the model.  Requires `"nlocations_output"`, `"nlocations_input"` and
`"nlags"` in `settings`; reads optional `model_pars` `"a"` and `"mod_activation"`.
"""
function BiLinearSurgeInteractionModel(settings::Dict{String, Any})
    nstations = settings["nlocations_output"]
    nwind     = settings["nlocations_input"]
    nlags     = settings["nlags"]

    mp    = get(settings, "model_pars", Dict{String, Any}())
    a     = Float32(get(mp, "a", 0.1))
    σ     = _mod_activation(get(mp, "mod_activation", "identity"))

    surge = Dense(3 * nwind * nlags => nstations)
    V     = zeros(Float32, nstations, nlags)   # zero-init → modulation ≡ 1 (LinearSurge start)
    flux  = BiLinearSurgeInteractionFlux(surge, StationTideModulation(V, a, σ))
    return BiLinearSurgeInteractionModel(flux, settings)
end

get_flux_model(m::BiLinearSurgeInteractionModel) = m.flux_model
get_settings(m::BiLinearSurgeInteractionModel)   = m.settings

# preprocess (both forms), postprocess!, forward, train_model!, save/load are all
# inherited from AbstractSurgeInteractionModel / AbstractSurgeModel / AbstractFluxModel.
