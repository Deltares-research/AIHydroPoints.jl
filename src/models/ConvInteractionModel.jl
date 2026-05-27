# ConvInteractionModel.jl
#
# Concrete subtype of AbstractInteractionModel.  Uses tide and surge as two
# input channels and applies strided Conv1D layers over the lag dimension.
# No per-station gating — simpler and faster than ProductInteractionModel.
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
# ConvInteractionFlux — Flux model struct
# ──────────────────────────────────────────────────────────────────────────────

"""
    struct ConvInteractionFlux{T}

Flux model for `ConvInteractionModel`.  Applies a chain of strided `Conv1D`
layers directly to the `(nlags, 2, nsamples)` tide+surge input (no station
gating).
"""
struct ConvInteractionFlux{T}
    conv_chain :: T   # Chain of Conv layers + Flux.flatten
end

function (l::ConvInteractionFlux)(x)
    _, x_ts = x        # ignore station encoding
    return l.conv_chain(x_ts)   # (1, nsamples)
end

Flux.@layer ConvInteractionFlux

"""
    ConvInteractionFlux(nlags, channels)

Construct the Flux model.

- `channels`: output-channel list for each Conv layer (last element must be 1).
  `nlags` must equal `2^length(channels)` so each stride-2 Conv halves the
  lag dimension until a scalar remains.
"""
function ConvInteractionFlux(nlags::Int, channels::Vector{Int})
    in_ch = [2; channels[1:end-1]]
    acts  = [fill(tanh, length(channels) - 1); [identity]]
    conv_chain = Chain(
        [Conv((2,), ic => oc, act, stride=(2,), pad=SamePad())
         for (ic, oc, act) in zip(in_ch, channels, acts)]...,
        Flux.flatten,
    )
    return ConvInteractionFlux(conv_chain)
end

# ──────────────────────────────────────────────────────────────────────────────
# ConvInteractionModel
# ──────────────────────────────────────────────────────────────────────────────

"""
    ConvInteractionModel <: AbstractInteractionModel

Interaction model that treats tide and surge as two input channels and applies
strided `Conv1D` layers over the lag dimension.  Simpler than
`ProductInteractionModel` — no per-station multiplicative gate.

`nlags` must equal `2^length(model_pars["channels"])`.

## Constructor

```julia
model = ConvInteractionModel(settings::Dict{String, Any})
```

Required keys in `settings`:

| Key | Description |
|---|---|
| `"nlocations_output"` | Number of tide/surge/waterlevel locations |
| `"nlags"`     | Number of lagged time steps |

Optional key:

| Key | Default | Description |
|---|---|---|
| `"model_pars"` | `Dict("channels" => [32, 16, 1])` | Conv output channel list |

## Data flow

```
preprocess → x_station (nstations, nstations * ntimes_valid)   [one-hot, ignored]
             x_ts      (nlags, 2, nstations * ntimes_valid)    [Z-scored lags]

forward    → Conv1D(stride=2, SamePad) × N → flatten
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

Construct a `ConvInteractionModel`.  Requires `"nlocations_output"` and `"nlags"` in
`settings`.  `nlags` must equal `2^length(model_pars["channels"])`.
"""
function ConvInteractionModel(settings::Dict{String, Any})
    nlags    = settings["nlags"]
    channels = get(settings, "model_pars", Dict("channels" => [32, 16, 1]))["channels"]

    @assert nlags == 2^length(channels) "nlags ($nlags) must equal 2^length(channels) ($(2^length(channels)))"

    flux = ConvInteractionFlux(nlags, channels)
    return ConvInteractionModel(flux, settings)
end

get_flux_model(m::ConvInteractionModel) = m.flux_model
get_settings(m::ConvInteractionModel)   = m.settings

"""
    forward(model::ConvInteractionModel, x::Tuple) -> Array{Float32, 3}

Unpack `(x_station, x_ts)`, run `x_ts` through the Conv chain (ignoring station
encoding), and return predictions reshaped to `(nstations, 1, ntimes_valid)`.
"""
function forward(model::ConvInteractionModel, x::Tuple)
    x_station, x_ts = x
    nstations = size(x_station, 1)
    y = model.flux_model((x_station, x_ts))   # (1, nstations * ntimes_valid)
    ntimes = size(y, 2) ÷ nstations
    return reshape(y, nstations, 1, ntimes)
end
