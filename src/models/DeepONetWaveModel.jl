# DeepONetWaveModel.jl
#
# Concrete subtype of AbstractWaveModel using a DeepONet-style dot-product merge:
# a station branch maps one-hot station vectors to a feature vector, and a branch
# network processes lagged wind stress via strided Conv1D layers; the two are merged
# via an inner product (dot product) to produce a scalar wave-height prediction.
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
# DeepONetWaveFlux — Flux model struct
# ──────────────────────────────────────────────────────────────────────────────

"""
    struct DeepONetWaveFlux{T1, T2}

Flux model for `DeepONetWaveModel`.

- `station_params`: `Dense(nstations → nchannel[end], relu; bias=false)` — maps
  the one-hot station vector to a feature vector of the same width as the branch
  network output.
- `branch_net`: chain of strided `Conv1D` layers that processes lagged wind-stress
  blocks `(nlags, 2*nwind, nsamples)` and flattens to `(nchannel[end], nsamples)`.

The forward pass computes `sum(station_feat .* branch_feat, dims=1)`, producing
a scalar `(1, nsamples)` for each (station, time) sample.
"""
struct DeepONetWaveFlux{T1, T2}
    station_params :: T1
    branch_net     :: T2
end

function (l::DeepONetWaveFlux)(x)
    x_station, x_input = x
    x1 = l.branch_net(x_input)          # (nchannel[end], nsamples)
    s1 = l.station_params(x_station)    # (nchannel[end], nsamples)
    return sum(s1 .* x1, dims=1)        # (1, nsamples)
end

@Flux.layer DeepONetWaveFlux

# ──────────────────────────────────────────────────────────────────────────────
# DeepONetWaveModel
# ──────────────────────────────────────────────────────────────────────────────

"""
    DeepONetWaveModel <: AbstractWaveModel

Wave model using a DeepONet-style dot-product merge between a learned station
feature vector and a convolutional branch that processes wind-stress history.

## Constructor

```julia
model = DeepONetWaveModel(settings::Dict{String, Any})
```

Required keys in `settings`:

| Key | Description |
|---|---|
| `"nlocations_output"` | Number of output locations |
| `"nlocations_input"`  | Number of input wind locations |
| `"nlags"`             | Number of lagged time steps (`= 2^length(nchannel)`) |

Optional keys (with defaults):

| Key | Default | Description |
|---|---|---|
| `"wind_scale"` | `0.5` | Divisor for wind stress |
| `"wave_scale"` | `3.0` | Divisor for wave height targets |
| `"model_pars"` | `Dict("nchannel"=>[32,32,32,16], "activation"=>"swish")` | Architecture |

## Data flow

```
preprocess → x_station (nstations, nstations * ntimes_valid)   [one-hot]
             x_input   (nlags, 2*nwind, nstations * ntimes_valid)

forward    → branch_net: Conv1D(stride=2) × N → flatten → (nchannel[end], nsamples)
             station_params: Dense → (nchannel[end], nsamples)
             dot product: sum(s .* b, dims=1) → (1, nsamples)
             reshape → (nstations, 1, ntimes_valid)

postprocess! → output["wave_height"].values .= y[:, 1, :] .* wave_scale
```
"""
mutable struct DeepONetWaveModel <: AbstractWaveModel
    flux_model :: DeepONetWaveFlux
    settings   :: Dict{String, Any}
end

"""
    DeepONetWaveModel(settings::Dict{String, Any}) -> DeepONetWaveModel

Construct a `DeepONetWaveModel`. Requires `"nlocations_output"`, `"nlocations_input"`, `"nlags"`.
`nlags` must equal `2^length(nchannel)`.
"""
function DeepONetWaveModel(settings::Dict{String, Any})
    nstations = settings["nlocations_output"]
    nwind     = settings["nlocations_input"]
    nlags     = settings["nlags"]
    mp        = get(settings, "model_pars",
                    Dict("nchannel" => [32, 32, 32, 16], "activation" => "swish"))
    nchannel  = mp["nchannel"]
    act_name  = get(mp, "activation", "swish")
    f_act     = act_name == "relu" ? relu : swish

    @assert nlags == 2^length(nchannel) "nlags ($nlags) must equal 2^length(nchannel) ($(2^length(nchannel)))"

    in_ch  = [2 * nwind; nchannel[1:end-1]]
    out_ch = nchannel
    acts   = [fill(f_act, length(nchannel) - 1); [identity]]

    branch_net = Chain(
        [Conv((2,), ic => oc, act, stride = (2,))
         for (ic, oc, act) in zip(in_ch, out_ch, acts)]...,
        Flux.flatten,
    )
    station_params = Dense(nstations => nchannel[end], relu; bias = false)

    return DeepONetWaveModel(DeepONetWaveFlux(station_params, branch_net), settings)
end

get_flux_model(m::DeepONetWaveModel) = m.flux_model
get_settings(m::DeepONetWaveModel)   = m.settings

# forward, postprocess!, and train_model! are inherited from AbstractWaveModel.
