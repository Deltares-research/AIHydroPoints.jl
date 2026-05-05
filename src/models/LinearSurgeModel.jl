# LinearSurgeModel.jl
#
# Concrete subtype of AbstractFluxModel for a linear surge model.
# Uses a single Flux.Dense layer (identity activation) to predict storm surge
# at nstations output locations from wind-stress and pressure forcing at nwind
# input locations over nlags previous time steps.
#
# Purpose: prototype / sanity-check for the AbstractFluxModel interface.
# It is intentionally minimal — no station encoding, no normalisation.

using Flux
using JLD2
using Printf: @sprintf
using ProgressMeter: Progress, next!

# ──────────────────────────────────────────────────────────────────────────────
# Struct
# ──────────────────────────────────────────────────────────────────────────────

"""
    LinearSurgeModel <: AbstractFluxModel

A minimal linear surge model: a single `Dense` layer (identity activation) that
maps flattened wind-stress and pressure history to storm-surge predictions.

## Constructor

```julia
model = LinearSurgeModel(settings::Dict{String, Any})
```

Required keys in `settings`: `"nstations"`, `"nwind"`, `"nlags"`.

Optional keys populated automatically by `train_model!` if absent:
`"out_names"`, `"out_lons"`, `"out_lats"`, `"out_quantity"`.
These must be present before calling `predict`.

## Input variables (`input` dict)

| Key | Shape | Description |
|---|---|---|
| `"wind_x"`   | `(nwind, T)` | East wind-stress component |
| `"wind_y"`   | `(nwind, T)` | North wind-stress component |
| `"pressure"` | `(nwind, T)` | Sea-level pressure (scaled: `2e-4*(p - 1e5)`) |

## Tensor layout

`preprocess` returns `(tensor, output)` where:
- `tensor` has shape `(1, 3*nwind, nlags, ntimes_valid)`.
- `output` is `Dict("surge" => ts)` with a zero-initialised `TimeSeries`
  whose metadata comes from `model.settings`.

`forward` flattens to `(3*nwind*nlags, ntimes_valid)`, applies `Dense`, and
reshapes to `(nstations, 1, ntimes_valid)`.

`postprocess!` writes `y[:, 1, :]` into `output["surge"].values` in-place.
"""
mutable struct LinearSurgeModel <: AbstractFluxModel
    flux_model
    settings :: Dict{String, Any}
end

"""
    LinearSurgeModel(settings::Dict{String, Any}) -> LinearSurgeModel

Construct an uninitialised `LinearSurgeModel` from `settings`.

Required keys: `"nstations"` (Int), `"nwind"` (Int), `"nlags"` (Int).
"""
function LinearSurgeModel(settings::Dict{String, Any})
    nstations = settings["nstations"]
    nwind     = settings["nwind"]
    nlags     = settings["nlags"]
    chain     = Dense(3 * nwind * nlags => nstations)   # linear (identity) output
    return LinearSurgeModel(chain, settings)
end

# ──────────────────────────────────────────────────────────────────────────────
# Required interface: get_flux_model, get_settings
# ──────────────────────────────────────────────────────────────────────────────

get_flux_model(m::LinearSurgeModel) = m.flux_model
get_settings(m::LinearSurgeModel)   = m.settings

# ──────────────────────────────────────────────────────────────────────────────
# preprocess
# ──────────────────────────────────────────────────────────────────────────────

"""
    preprocess(model::LinearSurgeModel, input::Dict{String, TimeSeries})
        -> (Array{Float32, 4}, Dict{String, TimeSeries})

Assemble a lag-window tensor from wind-stress and pressure forcing, and
pre-allocate the output `TimeSeries` for surge.

Returns `(tensor, output)` where:
- `tensor` has shape `(1, 3*nwind, nlags, ntimes_valid)`,
  with `ntimes_valid = length(times) - nlags + 1`.
- `output` is `Dict("surge" => ts)` with `ts.values` initialised to zeros.
  Station metadata (names, lons, lats) is read from `model.settings`.

Pressure is scaled by `2e-4*(p - 1e5)` to match the order of magnitude of the
wind-stress components (same convention as `surge.jl`).

Requires `"out_names"`, `"out_lons"`, `"out_lats"` to be present in
`model.settings`.  These are set automatically by `train_model!` on first use.
"""
function preprocess(model::LinearSurgeModel, input::Dict{String, TimeSeries})
    nwind     = model.settings["nwind"]
    nlags     = model.settings["nlags"]
    nstations = model.settings["nstations"]

    wind_x = Float32.(get_values(input["wind_x"]))
    wind_y = Float32.(get_values(input["wind_y"]))
    press  = Float32.(2e-4 .* (get_values(input["pressure"]) .- 1e5))

    times       = get_times(input["wind_x"])
    ntimes      = length(times)
    valid_range = nlags:ntimes
    nvalid      = length(valid_range)

    # Assemble lag windows: (3*nwind, nlags, nvalid)
    x = zeros(Float32, 3 * nwind, nlags, nvalid)
    for (i, t) in enumerate(valid_range)
        x[1:nwind,           :, i] = wind_x[:, t-nlags+1:t]
        x[nwind+1:2*nwind,   :, i] = wind_y[:, t-nlags+1:t]
        x[2*nwind+1:3*nwind, :, i] = press[ :, t-nlags+1:t]
    end

    # Pre-allocate output TimeSeries — metadata from settings
    out_ts = TimeSeries(
        zeros(Float32, nstations, nvalid),
        times[valid_range],
        model.settings["out_names"],
        model.settings["out_lons"],
        model.settings["out_lats"],
        get(model.settings, "out_quantity", "surge"),
        "LinearSurgeModel",
    )
    output = Dict{String, TimeSeries}("surge" => out_ts)

    tensor = reshape(x, 1, 3 * nwind, nlags, nvalid)
    return tensor, output
end

# ──────────────────────────────────────────────────────────────────────────────
# forward
# ──────────────────────────────────────────────────────────────────────────────

"""
    forward(model::LinearSurgeModel, x::Array{Float32, 4})
        -> Array{Float32, 3}

Flatten the last three dimensions to `(3*nwind*nlags, ntimes)`, apply the
`Dense` layer, and reshape to `(nstations, 1, ntimes)`.
"""
function forward(model::LinearSurgeModel, x::Array{Float32, 4})
    _, nfeatures, nlags_dim, ntimes = size(x)
    x_flat = reshape(x, nfeatures * nlags_dim, ntimes)  # (3*nwind*nlags, ntimes)
    y      = model.flux_model(x_flat)                    # (nstations, ntimes)
    return reshape(y, size(y, 1), 1, ntimes)             # (nstations, 1, ntimes)
end

# ──────────────────────────────────────────────────────────────────────────────
# postprocess!
# ──────────────────────────────────────────────────────────────────────────────

"""
    postprocess!(output::Dict{String, TimeSeries}, model::LinearSurgeModel,
                 y::Array{Float32, 3})

Write the surge predictions from `y` into the pre-allocated `output["surge"]`
in-place.  `y` has shape `(nstations, 1, ntimes)`; the singleton feature
dimension is dropped before writing.
"""
function postprocess!(output::Dict{String, TimeSeries}, model::LinearSurgeModel,
                      y::Array{Float32, 3})
    output["surge"].values .= y[:, 1, :]
end

# ──────────────────────────────────────────────────────────────────────────────
# train_model!
# ──────────────────────────────────────────────────────────────────────────────

"""
    train_model!(model::LinearSurgeModel, train_settings::TrainingSettings,
                 input::Dict{String, TimeSeries}, target::Dict{String, TimeSeries})
        -> Vector{Float32}

Train the `Dense` layer in-place using minibatch gradient descent (Adam).

`input` must contain `"wind_x"`, `"wind_y"`, and `"pressure"`.
`target` must contain one variable (the surge ground truth); all output
variables are assumed to share the same station locations.

On first call, `"out_names"`, `"out_lons"`, `"out_lats"`, and `"out_quantity"`
are added to `model.settings` from the first `TimeSeries` in `target`.

If `train_settings.validation_split > 0`, the last fraction of the time series
is held out as a validation set and its RMSE is shown in the progress bar.

Returns a `Vector{Float32}` of per-epoch RMSE training losses.
"""
function train_model!(model::LinearSurgeModel, train_settings::TrainingSettings,
                      input::Dict{String, TimeSeries}, target::Dict{String, TimeSeries})

    # Populate output metadata from target if not yet present
    if !haskey(model.settings, "out_names")
        ts_ref = first(values(target))
        model.settings["out_names"]    = get_names(ts_ref)
        model.settings["out_lons"]     = Float64.(get_longitudes(ts_ref))
        model.settings["out_lats"]     = Float64.(get_latitudes(ts_ref))
        model.settings["out_quantity"] = get_quantity(ts_ref)
    end

    # Build full input tensor and target matrix
    tensor, _ = preprocess(model, input)
    _, nfeatures, nlags_dim, nfull = size(tensor)
    x_all = reshape(tensor, nfeatures * nlags_dim, nfull)  # (3*nwind*nlags, nfull)

    nlags     = model.settings["nlags"]
    ts_target = first(values(target))
    y_all = Float32.(get_values(ts_target))[:, nlags:end]  # (nstations, nfull)

    # Temporal train/validation split
    n_val   = round(Int, train_settings.validation_split * nfull)
    has_val = n_val > 0
    n_train = nfull - n_val
    x = x_all[:, 1:n_train]
    y = y_all[:, 1:n_train]
    x_val = has_val ? x_all[:, n_train+1:end] : nothing
    y_val = has_val ? y_all[:, n_train+1:end] : nothing

    # Training loop
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
            Flux.mse(m(x[:, idx]), y[:, idx])
        end
        Flux.update!(opt_state, flux_model, grads[1])

        train_rmse = sqrt(Flux.mse(flux_model(x), y))
        push!(train_losses, train_rmse)

        empty!(showvalues)
        push!(showvalues, "train RMSE" => @sprintf("%.4f", train_rmse))
        if has_val
            val_rmse = sqrt(Flux.mse(flux_model(x_val), y_val))
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
