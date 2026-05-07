# ConvWaveModel.jl
#
# Concrete subtype of AbstractWaveModel using a convolutional architecture.
# Also defines WaveInputLayer, which is referenced by the legacy create_wave_model
# in waves.jl.
#
# Inherits from AbstractWaveModel:
#   preprocess, postprocess!, train_model!, plot_series
#
# Inherits from AbstractFluxModel:
#   predict, save_params, load_params!
#
# Implements:
#   get_flux_model, get_settings, forward

using Flux

# ──────────────────────────────────────────────────────────────────────────────
# WaveInputLayer — combines one-hot station branch with a Conv1D wind branch
# ──────────────────────────────────────────────────────────────────────────────

"""
    struct WaveInputLayer{T1,T2}

Input layer for wave models.  Combines a station-modulation branch with a
1-D convolutional branch.  The station branch learns a per-station channel-wise
sensitivity profile (applied via `exp`), modulating the convolved wind features.
"""
struct WaveInputLayer{T1, T2}
    station_params :: T1   # Dense(nstations => nlags*nchannels, identity; bias=false)
    first_conv     :: T2   # Conv((1,), 2*nwind => nchannels)
end

"""
    WaveInputLayer(nstations, nlags, npars, nchannels, f_activation)

Construct a `WaveInputLayer`.

# Arguments

- `nstations`: Number of output (wave) stations.
- `nlags`: Number of input time lags.
- `npars`: Number of input features per time step (`2 * nwind`).
- `nchannels`: Number of output channels.
- `f_activation`: Activation for the convolutional branch.
"""
WaveInputLayer(nstations, nlags, npars, nchannels, f_activation) = WaveInputLayer(
    Dense(nstations => (nlags * nchannels), identity; bias=false),
    Conv((1,), npars => nchannels, f_activation),
)

function (l::WaveInputLayer)(x)
    x_station, x_input = x
    x1 = l.first_conv(x_input)          # (nlags, nchannels, nsamples)
    s1 = l.station_params(x_station)    # (nlags*nchannels, nsamples)
    s1 = reshape(s1, size(x1))          # (nlags, nchannels, nsamples)
    return exp.(s1) .* x1
end

Flux.@layer WaveInputLayer

# ──────────────────────────────────────────────────────────────────────────────
# ConvWaveModel
# ──────────────────────────────────────────────────────────────────────────────

"""
    ConvWaveModel <: AbstractWaveModel

Wave model using `WaveInputLayer` followed by strided `Conv1D` layers.
`nlags` must equal `2^length(model_pars["nchannel"])`.

## Constructor

```julia
model = ConvWaveModel(settings::Dict{String, Any})
```

Required keys in `settings`:

| Key | Description |
|---|---|
| `"nstations"` | Number of output stations |
| `"nwind"`     | Number of input wind stations |
| `"nlags"`     | Number of lagged time steps |

Optional keys (with defaults):

| Key | Default | Description |
|---|---|---|
| `"wind_scale"` | `0.5` | Divisor for wind stress |
| `"wave_scale"` | `3.0` | Divisor for wave height targets |
| `"n_input_channels"` | `64` | Channels in WaveInputLayer |
| `"model_pars"` | `Dict("nchannel"=>[64,64,64,1], "activation"=>"swish")` | Architecture |

## Data flow

```
preprocess → x_station (nstations, nstations * ntimes_valid)   [one-hot]
             x_input   (nlags, 2*nwind, nstations * ntimes_valid) [wind stress lags]

forward    → WaveInputLayer → Conv1D(stride=2) × N → flatten
          → (1, nstations * ntimes_valid) → reshape (nstations, 1, ntimes_valid)

postprocess! → output["wave_height"].values .= y[:, 1, :] .* wave_scale
```
"""
mutable struct ConvWaveModel <: AbstractWaveModel
    flux_model
    settings :: Dict{String, Any}
end

"""
    ConvWaveModel(settings::Dict{String, Any}) -> ConvWaveModel

Construct a `ConvWaveModel`.  Requires `"nstations"`, `"nwind"`, `"nlags"` in
`settings`.  `nlags` must equal `2^length(nchannel)`.
"""
function ConvWaveModel(settings::Dict{String, Any})
    nstations = settings["nstations"]
    nwind     = settings["nwind"]
    nlags     = settings["nlags"]
    n_ch      = get(settings, "n_input_channels", 64)
    mp        = get(settings, "model_pars",
                    Dict("nchannel" => [64, 64, 64, 1], "activation" => "swish"))
    nchannel  = mp["nchannel"]
    act_name  = get(mp, "activation", "swish")
    f_act     = act_name == "relu" ? relu : swish

    @assert nlags == 2^length(nchannel) "nlags ($nlags) must equal 2^length(nchannel) ($(2^length(nchannel)))"

    in_ch = [n_ch; nchannel[1:end-1]]
    out_ch = nchannel
    acts   = [fill(f_act, length(nchannel) - 1); [identity]]

    chain = Chain(
        WaveInputLayer(nstations, nlags, 2 * nwind, n_ch, f_act),
        [Conv((2,), ic => oc, act, stride = (2,))
         for (ic, oc, act) in zip(in_ch, out_ch, acts)]...,
        Flux.flatten,
    )

    return ConvWaveModel(chain, settings)
end

get_flux_model(m::ConvWaveModel) = m.flux_model
get_settings(m::ConvWaveModel)   = m.settings

"""
    forward(model::ConvWaveModel, x::Tuple) -> Array{Float32, 3}

Unpack `(x_station, x_input)`, run through the Conv chain, and return
predictions reshaped to `(nstations, 1, ntimes_valid)`.
"""
function forward(model::ConvWaveModel, x::Tuple)
    x_station, x_input = x
    nstations = size(x_station, 1)
    y = model.flux_model((x_station, x_input))   # (1, nstations * ntimes_valid)
    ntimes = size(y, 2) ÷ nstations
    return reshape(y, nstations, 1, ntimes)
end
