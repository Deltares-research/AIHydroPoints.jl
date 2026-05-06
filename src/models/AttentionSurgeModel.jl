# AttentionSurgeModel.jl
#
# Concrete subtype of AbstractSurgeModel using the attention-based surge
# architecture (branch/trunk/downsample with graph adjacency).
#
# Inherits from AbstractSurgeModel:
#   postprocess!, save_params, load_params!, predict
#
# Overrides (different tensor structure — two inputs instead of one):
#   preprocess, forward, train_model!

using Flux
using Dates
using Printf: @sprintf
using ProgressMeter: Progress, next!

# ──────────────────────────────────────────────────────────────────────────────
# Internal Flux architecture
# ──────────────────────────────────────────────────────────────────────────────

"""
    AttentionSurgeFlux

Internal Flux model for `AttentionSurgeModel`. Combines a transformer branch
network (processing wind/pressure history) with a trunk network (processing
station metadata) via a graph-adjacency-weighted merge.

Not exported — construct via `AttentionSurgeModel`.
"""
struct AttentionSurgeFlux{P, Q, R, T}
    branch_net :: P
    trunk_net  :: Q
    downsample :: R
    adjacency  :: AbstractArray{T, 2}
end

function (m::AttentionSurgeFlux)(x_station, x_wind)
    nbatch     = size(x_station)[end]
    branch_out = m.branch_net(x_wind)
    trunk_out  = m.trunk_net(x_station)
    nwind      = size(branch_out, 1)
    merged = batched_mul(
        batched_transpose(trunk_out .* m.adjacency),
        reshape(branch_out, (nwind, :, nbatch)),
    )
    return m.downsample(merged)
end

@Flux.layer AttentionSurgeFlux

# ──────────────────────────────────────────────────────────────────────────────
# AttentionSurgeModel struct and constructor
# ──────────────────────────────────────────────────────────────────────────────

"""
    AttentionSurgeModel <: AbstractSurgeModel

Surge model using a transformer branch network for wind/pressure history and a
dense trunk network for station metadata, merged via graph adjacency weights.

## Constructor

```julia
model = AttentionSurgeModel(settings::Dict{String, Any}, gn::GraphNetwork)
```

Required keys in `settings`: `"nstations"`, `"nwind"`, `"nlags"`, `"model_pars"`.

Required keys in `"model_pars"`:
- `"nembed"`, `"theta"`, `"nheads"`, `"nlayers_branch"`, `"nlayers_trunk"`, `"nhidden_trunk"`

## Input variables

Same as `AbstractSurgeModel`: `"wind_x"`, `"wind_y"`, `"pressure"`.
Station encoding (lat/lon, time) is built automatically in `preprocess` from
`model.settings` and the input times.
"""
mutable struct AttentionSurgeModel <: AbstractSurgeModel
    flux_model :: AttentionSurgeFlux
    settings   :: Dict{String, Any}
end

"""
    AttentionSurgeModel(settings::Dict{String, Any}, gn::GraphNetwork)
        -> AttentionSurgeModel

Construct an `AttentionSurgeModel` from `settings` and a `GraphNetwork` for
the spatial adjacency structure.
"""
function AttentionSurgeModel(settings::Dict{String, Any}, gn::GraphNetwork)
    nlags   = settings["nlags"]
    nwind   = settings["nwind"]
    mp      = settings["model_pars"]
    nembed          = mp["nembed"]
    theta           = mp["theta"]
    nheads          = mp["nheads"]
    nlayers_branch  = mp["nlayers_branch"]
    nlayers_trunk   = mp["nlayers_trunk"]
    nhidden_trunk   = mp["nhidden_trunk"]

    embed     = Embedder(3 * nwind, nembed)
    deembed   = Deembedder(embed)
    pos_embed = SinCosPosEmbedder(nembed, nlags; theta)

    branch_net = Chain(
        embed,
        pos_embed,
        [Transformer(nembed, nheads) for _ in 1:nlayers_branch]...,
        deembed,
        x -> reshape(x, (nwind, 3, nlags, :)),
    )

    trunk_net = Chain(
        Dense(6 => nhidden_trunk),
        [Dense(nhidden_trunk => nhidden_trunk) for _ in 1:nlayers_trunk]...,
        Dense(nhidden_trunk => nwind),
    )

    downsample = Conv((1,), 3 * nlags => nlags, identity; stride=(1,), pad=SamePad())

    flux_model = AttentionSurgeFlux(branch_net, trunk_net, downsample, gn.adjacency)
    return AttentionSurgeModel(flux_model, settings)
end

get_flux_model(m::AttentionSurgeModel) = m.flux_model
get_settings(m::AttentionSurgeModel)   = m.settings

# ──────────────────────────────────────────────────────────────────────────────
# preprocess — overrides AbstractSurgeModel (two inputs, not one)
# ──────────────────────────────────────────────────────────────────────────────

"""
    preprocess(model::AttentionSurgeModel, input::Dict{String, TimeSeries})
        -> (Tuple, Dict{String, TimeSeries})

Build wind/pressure lag tensor and station-encoding tensor, and pre-allocate
the output `TimeSeries` for surge.

Returns `((x_station, x_wind), output)` where:
- `x_station` has shape `(6, nstations, ntimes_valid)`: cos/sin of lat, lon,
  and day-of-year for each output station at each valid time step.
- `x_wind` has shape `(3*nwind, nlags, ntimes_valid)`: lagged wind-stress and
  scaled pressure history.

Accepts either `"stress_x"`/`"stress_y"` (used directly) or `"wind_x"`/`"wind_y"`
(converted via `uv_to_stress_xy`).

Requires `"out_lats"`, `"out_lons"` to be present in `model.settings` (set
automatically by `train_model!` on first call).
"""
function preprocess(model::AttentionSurgeModel, input::Dict{String, TimeSeries})
    settings  = get_settings(model)
    nwind     = settings["nwind"]
    nlags     = settings["nlags"]
    nstations = settings["nstations"]

    stress_x, stress_y = _get_stress(input)
    press  = Float32.(2e-4 .* (get_values(input["pressure"]) .- 1e5))

    times       = get_times(input[_wind_key(input)])
    ntimes      = length(times)
    valid_range = nlags:ntimes
    nvalid      = length(valid_range)

    # Stress/pressure lag tensor: (3*nwind, nlags, nvalid)
    x_wind = zeros(Float32, 3 * nwind, nlags, nvalid)
    for (i, t) in enumerate(valid_range)
        x_wind[1:nwind,           :, i] = stress_x[:, t-nlags+1:t]
        x_wind[nwind+1:2*nwind,   :, i] = stress_y[:, t-nlags+1:t]
        x_wind[2*nwind+1:3*nwind, :, i] = press[   :, t-nlags+1:t]
    end

    # Station encoding: (6, nstations, nvalid)
    lats       = settings["out_lats"]
    lons       = settings["out_lons"]
    dayperiod  = 365.25
    times_day  = Dates.dayofyear.(times[valid_range])
    times_cos  = Float32.(cos.(2π .* times_day ./ dayperiod))'  # (1, nvalid)
    times_sin  = Float32.(sin.(2π .* times_day ./ dayperiod))'

    x_station = zeros(Float32, 6, nstations, nvalid)
    x_station[1, :, :] .= Float32.(cos.(deg2rad.(lats)))
    x_station[2, :, :] .= Float32.(sin.(deg2rad.(lats)))
    x_station[3, :, :] .= Float32.(cos.(deg2rad.(lons)))
    x_station[4, :, :] .= Float32.(sin.(deg2rad.(lons)))
    x_station[5, :, :] .= times_cos
    x_station[6, :, :] .= times_sin

    # Pre-allocate output TimeSeries
    out_ts = TimeSeries(
        zeros(Float32, nstations, nvalid),
        times[valid_range],
        settings["out_names"],
        settings["out_lons"],
        settings["out_lats"],
        get(settings, "out_quantity", "surge"),
        "AttentionSurgeModel",
    )
    output = Dict{String, TimeSeries}("surge" => out_ts)

    return (x_station, x_wind), output
end

# ──────────────────────────────────────────────────────────────────────────────
# forward — overrides AbstractSurgeModel (unpacks tuple input)
# ──────────────────────────────────────────────────────────────────────────────

"""
    forward(model::AttentionSurgeModel, x::Tuple) -> Array{Float32, 3}

Unpack `(x_station, x_wind)` from `x`, run `AttentionSurgeFlux`, and return
the last-lag slice reshaped to `(nstations, 1, ntimes)`.
"""
function forward(model::AttentionSurgeModel, x::Tuple)
    x_station, x_wind = x
    y = model.flux_model(x_station, x_wind)   # (nstations, nlags, ntimes)
    return reshape(y[:, end, :], size(y, 1), 1, size(y, 3))
end

# ──────────────────────────────────────────────────────────────────────────────
# train_model! — overrides AbstractSurgeModel (batches over time axis as tuple)
# ──────────────────────────────────────────────────────────────────────────────

"""
    train_model!(model::AttentionSurgeModel, train_settings::TrainingSettings,
                 input::Dict{String, TimeSeries}, target::Dict{String, TimeSeries})
        -> (Vector{Float32}, Vector{Float32})

Train `AttentionSurgeModel` in-place using minibatch gradient descent (Adam).
Batches are sampled randomly from the time axis.

See `AbstractSurgeModel` docstring for argument conventions and return value.
"""
function train_model!(model::AttentionSurgeModel, train_settings::TrainingSettings,
                      input::Dict{String, TimeSeries}, target::Dict{String, TimeSeries})

    settings = get_settings(model)

    # Populate output metadata from target if not yet present
    if !haskey(settings, "out_names")
        ts_ref = first(values(target))
        settings["out_names"]    = get_names(ts_ref)
        settings["out_lons"]     = Float64.(get_longitudes(ts_ref))
        settings["out_lats"]     = Float64.(get_latitudes(ts_ref))
        settings["out_quantity"] = get_quantity(ts_ref)
    end

    (x_station, x_wind), _ = preprocess(model, input)
    nvalid = size(x_wind, 3)

    nlags     = settings["nlags"]
    ts_target = first(values(target))
    y_all = Float32.(get_values(ts_target))[:, nlags:end]   # (nstations, nvalid)

    # Temporal train/validation split
    n_val   = round(Int, train_settings.validation_split * nvalid)
    has_val = n_val > 0
    n_train = nvalid - n_val

    x_st  = x_station[:, :, 1:n_train];  x_st_val  = has_val ? x_station[:, :, n_train+1:end] : nothing
    x_w   = x_wind[:, :, 1:n_train];     x_w_val   = has_val ? x_wind[:, :, n_train+1:end]    : nothing
    y     = y_all[:, 1:n_train];          y_val     = has_val ? y_all[:, n_train+1:end]         : nothing

    flux_model = get_flux_model(model)
    opt_state  = Flux.setup(Adam(train_settings.learning_rate), flux_model)
    nbatch     = min(train_settings.nbatches, n_train)

    train_losses = Float32[]
    val_losses   = Float32[]
    showvalues   = Pair{String,String}[]
    progress     = Progress(train_settings.nepochs; desc="Training: ", showspeed=true)
    log_every    = max(1, train_settings.nepochs ÷ 10)

    for epoch in 1:train_settings.nepochs
        idx = sortperm(rand(n_train))[1:nbatch]
        _, grads = Flux.withgradient(flux_model) do m
            Flux.mse(m(x_st[:, :, idx], x_w[:, :, idx])[:, end, :], y[:, idx])
        end
        Flux.update!(opt_state, flux_model, grads[1])

        train_rmse = sqrt(Flux.mse(flux_model(x_st, x_w)[:, end, :], y))
        push!(train_losses, train_rmse)

        empty!(showvalues)
        push!(showvalues, "train RMSE" => @sprintf("%.4f", train_rmse))
        if has_val
            val_rmse = sqrt(Flux.mse(flux_model(x_st_val, x_w_val)[:, end, :], y_val))
            push!(val_losses, val_rmse)
            push!(showvalues, "val RMSE  " => @sprintf("%.4f", val_rmse))
        end
        next!(progress; showvalues)

        if epoch % log_every == 0 || epoch == train_settings.nepochs
            msg = @sprintf("epoch %d/%d  train RMSE: %.4f", epoch, train_settings.nepochs, train_rmse)
            has_val && (msg *= @sprintf("  val RMSE: %.4f", val_losses[end]))
            @info msg
        end
    end

    return train_losses, val_losses
end
