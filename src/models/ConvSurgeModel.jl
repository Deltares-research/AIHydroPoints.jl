# ConvSurgeModel.jl
#
# Concrete subtype of AbstractSurgeModel using Conv1D layers over the lag
# dimension to predict storm surge from wind-stress and pressure history.
#
# Inherits from AbstractSurgeModel without override:
#   preprocess, postprocess!, train_model!, plot_series, save_params, load_params!

using Flux

"""
    ConvSurgeModel <: AbstractSurgeModel

Surge model that applies 1-D convolutions over the lag dimension of the
wind-stress and pressure history to predict storm surge.

## Constructor

```julia
model = ConvSurgeModel(settings::Dict{String, Any})
```

Required keys in `settings`: `"nlocations_output"`, `"nlocations_input"`, `"nlags"`.

Optional key `"model_pars"` (Dict):
- `"channels"` — output channels for each Conv1D layer (default `[32, 16]`)
- `"filtersize"` — Conv1D kernel width **and stride** (default `3`); using
  `stride = filtersize` tiles the lag axis into non-overlapping windows
- `"activation"` — Conv1D activation, `"swish"` (default) or `"relu"`

## Architecture

```
(nlags, 3*nlocations_input, ntimes)         ← conv-ready tensor from preprocess
    Conv1D(filtersize, 3*nlocations_input → channels[1], act, stride=filtersize, SamePad)
    Conv1D(filtersize, channels[1] → channels[2], act, stride=filtersize, SamePad)
    ...
    flatten → (nlags_out * channels[end], ntimes)
    Dense(nlags_out * channels[end] → nlocations_output)
(nlocations_output, ntimes)
```

The 1-D convolution slides over the **lag** axis (axis 1); the
`3*nlocations_input` stress/pressure fields are the **channels** (axis 2); batch-
time is the batch axis (axis 3). `preprocess` builds this `(lag, channel, batch)`
layout directly, so — unlike the previous `reshape`-based version — memory order
and axis interpretation always agree (see `docs/notes_dimensions.md`, Note 2).

Each Conv1D layer uses `stride = filtersize` with `pad=SamePad()`, so its kernel
tiles the lag axis into **non-overlapping** windows — every lag point is visited
exactly once — and the lag length shrinks by a factor of `filtersize` per layer
(`cld(·, filtersize)`). This reduces the flattened `Dense` input from
`nlags * channels[end]` to `nlags_out * channels[end]` (where `nlags_out` is the
lag length after all conv layers), cutting the parameter count.
"""
mutable struct ConvSurgeModel <: AbstractSurgeModel
    flux_model
    settings :: Dict{String, Any}
end

"""
    ConvSurgeModel(settings::Dict{String, Any}) -> ConvSurgeModel

Construct a `ConvSurgeModel` from `settings`.

Required keys: `"nlocations_output"` (Int), `"nlocations_input"` (Int), `"nlags"` (Int).
"""
function ConvSurgeModel(settings::Dict{String, Any})
    nstations  = settings["nlocations_output"]
    nwind      = settings["nlocations_input"]
    nlags      = settings["nlags"]
    mp         = get(settings, "model_pars", Dict{String, Any}())
    channels   = get(mp, "channels",   [32, 16])
    filtersize = get(mp, "filtersize", 3)
    act_name   = get(mp, "activation", "swish")
    f_act      = act_name == "relu" ? relu : swish

    n_in    = 3 * nwind
    ch_seq  = [n_in; channels]

    # stride == filtersize tiles the lag axis into non-overlapping windows
    # (every lag visited once). With SamePad, each layer's lag length becomes
    # cld(prev, filtersize); track it through the stack to size the final Dense.
    nlags_out = nlags
    for _ in 1:length(channels)
        nlags_out = cld(nlags_out, filtersize)
    end

    # The chain consumes the conv-ready (lag, channel, batch) tensor from
    # preprocess directly — no internal reshape (that was the Note-2 bug).
    chain = Chain(
        [Conv((filtersize,), ch_seq[i] => ch_seq[i+1], f_act; stride=(filtersize,), pad=SamePad())
         for i in 1:length(ch_seq)-1]...,
        Flux.flatten,
        Dense(nlags_out * channels[end] => nstations, identity) #allow negative surge
    )
    return ConvSurgeModel(chain, settings)
end

get_flux_model(m::ConvSurgeModel) = m.flux_model
get_settings(m::ConvSurgeModel)   = m.settings

"""
    preprocess(model::ConvSurgeModel, input::Dict{String, TimeSeries})
        -> (Tuple, Dict{String, TimeSeries})

Assemble the conv-ready input tuple from the shared lag windows.

Returns `((x,), output)` where `x` has shape `(nlags, 3*nlocations_input,
ntimes_valid)`:

```
axis 1 → lag         (Δt) — the spatial axis the 1-D Conv slides over
axis 2 → channel     (stress_x block, stress_y block, pressure block)
axis 3 → batch-time  (valid step)
```

The tensor is built **directly** in this order by transposing each
`(point, lag, batch-time)` forcing window into `(lag, point, batch-time)` and
stacking along the channel axis. This is the honest replacement for the previous
`reshape(x, nlags, n_in, …)`, which reinterpreted a `(point·quantity)`-fastest
buffer as `lag`-fastest and silently scrambled the two axes whenever
`nlags ≠ 3*nlocations_input` (see `docs/notes_dimensions.md`, Note 2).

`forward` and `postprocess!` are inherited from `AbstractSurgeModel`.
"""
function preprocess(model::ConvSurgeModel, input::Dict{String, TimeSeries})
    # sx, sy, pr :: (nwind, nlags, nvalid)   — shared extraction
    sx, sy, pr, times_valid = _surge_lag_windows(model, input)
    nwind  = size(sx, 1)
    nlags  = size(sx, 2)
    nvalid = size(sx, 3)

    # Conv-ready layout (lag, channel, batch-time). permutedims materialises the
    # (point, lag, …) → (lag, point, …) transpose so memory matches the axes.
    x = zeros(Float32, nlags, 3 * nwind, nvalid)
    x[:, 1:nwind,           :] = permutedims(sx, (2, 1, 3))   # channels 1..nwind        = stress_x
    x[:, nwind+1:2*nwind,   :] = permutedims(sy, (2, 1, 3))   # channels nwind+1..2*nwind = stress_y
    x[:, 2*nwind+1:3*nwind, :] = permutedims(pr, (2, 1, 3))   # channels 2*nwind+1..3*nwind = pressure
    return (x,), _alloc_surge_output(model, times_valid)
end
