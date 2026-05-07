# ConvInteractionModel.jl
#
# Concrete subtype of AbstractInteractionModel.  Uses an InteractionInputLayer
# (per-station multiplicative gate on the tide+surge lags) followed by strided
# Conv1D layers that compress the lag dimension down to a scalar output.
#
# Inherits from AbstractInteractionModel:
#   preprocess, postprocess!, train_model!, plot_series
#
# Inherits from AbstractFluxModel:
#   predict, save_params, load_params!
#
# Implements:
#   get_flux_model, get_settings, forward

using Flux

# ──────────────────────────────────────────────────────────────────────────────
# InteractionInputLayer — per-station multiplicative gate on tide+surge lags
# ──────────────────────────────────────────────────────────────────────────────

"""
    struct InteractionInputLayer{T}

Input layer for interaction models.  Applies a per-station learned gate to the
`(nlags, 2, nsamples)` tide+surge input.  The station branch produces a
`(nlags, 2, nsamples)` sensitivity tensor via `Dense(nstations => nlags*2)`,
which is multiplied element-wise with the raw input.
"""
struct InteractionInputLayer{T}
    station_params :: T   # Dense(nstations => nlags*2, identity; bias=false)
end

"""
    InteractionInputLayer(nstations, nlags)

Construct an `InteractionInputLayer` for the given station count and lag length.
"""
InteractionInputLayer(nstations::Int, nlags::Int) = InteractionInputLayer(
    Dense(nstations => nlags * 2, identity; bias=false),
)

function (l::InteractionInputLayer)(x)
    x_station, x_ts = x
    nlags, npars, nbatch = size(x_ts)
    s1 = l.station_params(x_station)      # (nlags*2, nbatch)
    s1 = reshape(s1, nlags, npars, nbatch)
    return s1 .* x_ts
end

Flux.@layer InteractionInputLayer

# ──────────────────────────────────────────────────────────────────────────────
# ConvInteractionFlux — Flux model struct
# ──────────────────────────────────────────────────────────────────────────────

"""
    struct ConvInteractionFlux{T1, T2}

Flux model for `ConvInteractionModel`.  Combines an `InteractionInputLayer`
gate with a chain of strided `Conv1D` layers that collapse the lag dimension
to a scalar.
"""
struct ConvInteractionFlux{T1, T2}
    input_layer :: T1   # InteractionInputLayer
    conv_chain  :: T2   # Chain of Conv layers + Flux.flatten
end

function (l::ConvInteractionFlux)(x)
    z = l.input_layer(x)     # (nlags, 2, nsamples)
    return l.conv_chain(z)   # (1, nsamples)
end

Flux.@layer ConvInteractionFlux

"""
    ConvInteractionFlux(nstations, nlags, channels)

Construct the Flux model.

- `channels`: output-channel list for each Conv layer (last element must be 1).
  `nlags` must equal `2^length(channels)` so each stride-2 Conv halves the
  lag dimension until a scalar remains.
"""
function ConvInteractionFlux(nstations::Int, nlags::Int, channels::Vector{Int})
    in_ch = [2; channels[1:end-1]]
    acts  = [fill(tanh, length(channels) - 1); [identity]]
    conv_chain = Chain(
        [Conv((2,), ic => oc, act, stride=(2,), pad=SamePad())
         for (ic, oc, act) in zip(in_ch, channels, acts)]...,
        Flux.flatten,
    )
    return ConvInteractionFlux(InteractionInputLayer(nstations, nlags), conv_chain)
end

# ──────────────────────────────────────────────────────────────────────────────
# ConvInteractionModel
# ──────────────────────────────────────────────────────────────────────────────

"""
    ConvInteractionModel <: AbstractInteractionModel

Interaction model using `InteractionInputLayer` followed by strided `Conv1D`
layers.  `nlags` must equal `2^length(model_pars["channels"])`.

## Constructor

```julia
model = ConvInteractionModel(settings::Dict{String, Any})
```

Required keys in `settings`:

| Key | Description |
|---|---|
| `"nstations"` | Number of tide/surge/waterlevel stations |
| `"nlags"`     | Number of lagged time steps |

Optional key:

| Key | Default | Description |
|---|---|---|
| `"model_pars"` | `Dict("channels" => [32, 16, 1])` | Conv output channel list |

## Data flow

```
preprocess → x_station (nstations, nstations * ntimes_valid)   [one-hot]
             x_ts      (nlags, 2, nstations * ntimes_valid)    [Z-scored lags]

forward    → InteractionInputLayer → Conv1D(stride=2, SamePad) × N → flatten
          → (1, nstations * ntimes_valid) → reshape (nstations, 1, ntimes_valid)

postprocess! → output["waterlevel"].values .= y .* output_std .+ output_mu
```
"""
mutable struct ConvInteractionModel <: AbstractInteractionModel
    flux_model :: ConvInteractionFlux
    settings   :: Dict{String, Any}
end

"""
    ConvInteractionModel(settings::Dict{String, Any}) -> ConvInteractionModel

Construct a `ConvInteractionModel`.  Requires `"nstations"` and `"nlags"` in
`settings`.  `nlags` must equal `2^length(model_pars["channels"])`.
"""
function ConvInteractionModel(settings::Dict{String, Any})
    nstations = settings["nstations"]
    nlags     = settings["nlags"]
    channels  = get(settings, "model_pars", Dict("channels" => [32, 16, 1]))["channels"]

    @assert nlags == 2^length(channels) "nlags ($nlags) must equal 2^length(channels) ($(2^length(channels)))"

    flux = ConvInteractionFlux(nstations, nlags, channels)
    return ConvInteractionModel(flux, settings)
end

get_flux_model(m::ConvInteractionModel) = m.flux_model
get_settings(m::ConvInteractionModel)   = m.settings

"""
    forward(model::ConvInteractionModel, x::Tuple) -> Array{Float32, 3}

Unpack `(x_station, x_ts)`, run through the Conv chain, and return
predictions reshaped to `(nstations, 1, ntimes_valid)`.
"""
function forward(model::ConvInteractionModel, x::Tuple)
    x_station, x_ts = x
    nstations = size(x_station, 1)
    y = model.flux_model((x_station, x_ts))   # (1, nstations * ntimes_valid)
    ntimes = size(y, 2) ÷ nstations
    return reshape(y, nstations, 1, ntimes)
end
