# AbstractSurgeModel.jl
#
# Intermediate abstract type for all surge models.  Sits between AbstractFluxModel
# and concrete surge models (LinearSurgeModel, etc.) and implements the shared
# surge-specific logic: preprocess, postprocess!, and train_model!.
#
# Concrete subtypes must implement: get_flux_model, get_settings, forward.

using Flux
using Printf: @sprintf
using ProgressMeter: Progress, next!

# ──────────────────────────────────────────────────────────────────────────────
# Internal helper: get stress from input dict
# ──────────────────────────────────────────────────────────────────────────────

"""
    _get_stress(input::Dict{String, TimeSeries}) -> (Matrix{Float32}, Matrix{Float32})

Return `(stress_x, stress_y)` from `input`, converting from wind components if needed.

- If `input` contains `"stress_x"` / `"stress_y"`: used directly.
- If `input` contains `"wind_x"` / `"wind_y"`: converted via `uv_to_stress_xy`.
"""
function _get_stress(input::Dict{String, TimeSeries})
    if haskey(input, "stress_x")
        stress_x = Float32.(get_values(input["stress_x"]))
        stress_y = Float32.(get_values(input["stress_y"]))
    else
        raw_x = get_values(input["wind_x"])
        raw_y = get_values(input["wind_y"])
        stress_x = zeros(Float32, size(raw_x))
        stress_y = zeros(Float32, size(raw_y))
        for i in eachindex(raw_x)
            stress_x[i], stress_y[i] = uv_to_stress_xy(raw_x[i], raw_y[i])
        end
    end
    return stress_x, stress_y
end

"""
    _wind_key(input::Dict{String, TimeSeries}) -> String

Return the name of the wind/stress key present in `input` (`"stress_x"` or `"wind_x"`).
Used to extract times from the forcing TimeSeries.
"""
_wind_key(input::Dict{String, TimeSeries}) = haskey(input, "stress_x") ? "stress_x" : "wind_x"

"""
    AbstractSurgeModel <: AbstractFluxModel

Abstract supertype for surge models.  Provides shared implementations of
`preprocess`, `postprocess!`, and `train_model!` for models that predict
storm surge from wind-stress and pressure forcing at `nwind` locations over
`nlags` time steps.

## Required settings keys

| Key | Description |
|---|---|
| `"nlocations_output"` | Number of output (waterlevel) locations |
| `"nlocations_input"`  | Number of input (forcing) locations |
| `"nlags"`             | Number of lagged time steps used as input |

The following are populated automatically by `train_model!` on first call:
`"out_names"`, `"out_lons"`, `"out_lats"`, `"out_quantity"`.

## Tensor layout

`preprocess` produces a tensor of shape `(1, 3*nlocations_input, nlags, ntimes_valid)`,
where the three feature blocks are `wind_x`, `wind_y`, and scaled pressure.
`forward` must accept this 4-D tensor and return `(nlocations_output, 1, ntimes_valid)`.

## Concrete subtypes must implement

- `get_flux_model(m)` — return the underlying Flux chain
- `get_settings(m)` — return `Dict{String, Any}`
- `forward(m, x::Array{Float32,4}) -> Array{Float32,3}`
"""
abstract type AbstractSurgeModel <: AbstractFluxModel end

# ──────────────────────────────────────────────────────────────────────────────
# preprocess — shared across all surge models
# ──────────────────────────────────────────────────────────────────────────────

"""
    preprocess(model::AbstractSurgeModel, input::Dict{String, TimeSeries})
        -> (Array{Float32, 4}, Dict{String, TimeSeries})

Assemble a lag-window tensor from wind-stress and pressure forcing, and
pre-allocate the output `TimeSeries` for surge.

Returns `(tensor, output)` where:
- `tensor` has shape `(1, 3*nwind, nlags, ntimes_valid)`,
  with `ntimes_valid = length(times) - nlags + 1`.
- `output` is `Dict("surge" => ts)` with `ts.values` initialised to zeros.
  Station metadata is read from `model.settings`.

Accepts either `"stress_x"`/`"stress_y"` (used directly) or `"wind_x"`/`"wind_y"`
(converted via `uv_to_stress_xy`). Pressure is scaled by `2e-4*(p - 1e5)`.

Requires `"out_names"`, `"out_lons"`, `"out_lats"` to be present in
`model.settings`.  These are set automatically by `train_model!` on first use.
"""
function preprocess(model::AbstractSurgeModel, input::Dict{String, TimeSeries})
    settings  = get_settings(model)
    nwind     = settings["nlocations_input"]
    nlags     = settings["nlags"]
    nstations = settings["nlocations_output"]

    # Align input locations to training-time order (errors on missing, drops extras)
    if haskey(settings, "in_names")
        in_names = settings["in_names"]
        input = Dict(k => _check_and_align_locations(v, in_names, "input[\"$k\"]")
                     for (k, v) in input)
    end

    stress_x, stress_y = _get_stress(input)
    press  = Float32.(2e-4 .* (get_values(input["pressure"]) .- 1e5))

    times       = get_times(input[_wind_key(input)])
    ntimes      = length(times)
    valid_range = nlags:ntimes
    nvalid      = length(valid_range)

    x = zeros(Float32, 3 * nwind, nlags, nvalid)
    for (i, t) in enumerate(valid_range)
        x[1:nwind,           :, i] = stress_x[:, t-nlags+1:t]
        x[nwind+1:2*nwind,   :, i] = stress_y[:, t-nlags+1:t]
        x[2*nwind+1:3*nwind, :, i] = press[   :, t-nlags+1:t]
    end

    out_ts = TimeSeries(
        zeros(Float32, nstations, nvalid),
        times[valid_range],
        settings["out_names"],
        settings["out_lons"],
        settings["out_lats"],
        get(settings, "out_quantity", "surge"),
        string(typeof(model)),
    )
    output = Dict{String, TimeSeries}("surge" => out_ts)

    tensor = reshape(x, 1, 3 * nwind, nlags, nvalid)
    return tensor, output
end

# ──────────────────────────────────────────────────────────────────────────────
# postprocess! — shared across all surge models
# ──────────────────────────────────────────────────────────────────────────────

"""
    postprocess!(output::Dict{String, TimeSeries}, model::AbstractSurgeModel,
                 y::Array{Float32, 3})

Write surge predictions from `y` into `output["surge"]` in-place.
`y` has shape `(nstations, 1, ntimes)`; the singleton feature dimension is dropped.
"""
function postprocess!(output::Dict{String, TimeSeries}, model::AbstractSurgeModel,
                      y::Array{Float32, 3})
    output["surge"].values .= y[:, 1, :]
end

# ──────────────────────────────────────────────────────────────────────────────
# train_model! — shared across all surge models
# ──────────────────────────────────────────────────────────────────────────────

"""
    train_model!(model::AbstractSurgeModel, train_settings::TrainingSettings,
                 input::Dict{String, TimeSeries}, target::Dict{String, TimeSeries})
        -> (Vector{Float32}, Vector{Float32})

Train the model in-place using minibatch gradient descent (Adam).

`input` must contain `"wind_x"`, `"wind_y"`, and `"pressure"`.
`target` must contain one variable (the surge ground truth).

On first call, `"out_names"`, `"out_lons"`, `"out_lats"`, and `"out_quantity"`
are added to the model settings from the first `TimeSeries` in `target`.

If `train_settings.validation_split > 0`, the last fraction of the time series
is held out as a validation set and its RMSE is shown in the progress bar.

Returns `(train_losses, val_losses)` as `Vector{Float32}` per epoch.
`val_losses` is empty when `validation_split == 0` and no explicit validation
data is provided.

If `val_input` / `val_target` are supplied they are used directly and
`validation_split` is ignored.
"""
function train_model!(model::AbstractSurgeModel, train_settings::TrainingSettings,
                      input::Dict{String, TimeSeries}, target::Dict{String, TimeSeries};
                      val_input::Union{Dict{String,TimeSeries},Nothing}  = nothing,
                      val_target::Union{Dict{String,TimeSeries},Nothing} = nothing)

    settings = get_settings(model)

    # Populate output metadata from target if not yet present
    if !haskey(settings, "out_names")
        ts_ref = first(values(target))
        settings["out_names"]    = get_names(ts_ref)
        settings["out_lons"]     = Float64.(get_longitudes(ts_ref))
        settings["out_lats"]     = Float64.(get_latitudes(ts_ref))
        settings["out_quantity"] = get_quantity(ts_ref)
    end

    # Build full input tensor and target matrix
    tensor, _ = preprocess(model, input)
    _, nfeatures, nlags_dim, nfull = size(tensor)
    x_all = reshape(tensor, nfeatures * nlags_dim, nfull)

    nlags     = settings["nlags"]
    ts_target = first(values(target))
    y_all = Float32.(get_values(ts_target))[:, nlags:end]

    # Validation data: explicit split takes priority over validation_split
    if !isnothing(val_input)
        val_tensor, _ = preprocess(model, val_input)
        _, nf_v, nl_v, nv = size(val_tensor)
        x_val   = reshape(val_tensor, nf_v * nl_v, nv)
        y_val   = Float32.(get_values(first(values(val_target))))[:, nlags:end]
        has_val = true
        n_train = nfull
        x       = x_all
        y       = y_all
    else
        n_val   = round(Int, train_settings.validation_split * nfull)
        has_val = n_val > 0
        n_train = nfull - n_val
        x       = x_all[:, 1:n_train]
        y       = y_all[:, 1:n_train]
        x_val   = has_val ? x_all[:, n_train+1:end] : nothing
        y_val   = has_val ? y_all[:, n_train+1:end] : nothing
    end

    # Training loop
    flux_model = get_flux_model(model)
    opt_state  = Flux.setup(Adam(train_settings.learning_rate), flux_model)
    nbatch     = min(train_settings.nbatches, n_train)

    checkpoint_dir = get(settings, "model_dir", nothing)

    train_losses  = Float32[]
    val_losses    = Float32[]
    showvalues    = Pair{String,String}[]
    progress      = Progress(train_settings.nepochs; desc="Training: ", showspeed=true)
    log_every     = max(1, train_settings.nepochs ÷ 10)
    best_val_rmse = Inf32

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
            if !isnothing(checkpoint_dir) && val_rmse < best_val_rmse
                best_val_rmse = val_rmse
                save_params(model, joinpath(checkpoint_dir, "params_best.jld2"); overwrite=true)
            end
        end
        next!(progress; showvalues)

        if !isnothing(checkpoint_dir) && !isnothing(train_settings.checkpoints) &&
                epoch in train_settings.checkpoints
            save_params(model, joinpath(checkpoint_dir, "params_epoch_$(epoch).jld2"); overwrite=true)
        end

        if epoch % log_every == 0 || epoch == train_settings.nepochs
            msg = @sprintf("epoch %d/%d  train RMSE: %.4f", epoch, train_settings.nepochs, train_rmse)
            has_val && (msg *= @sprintf("  val RMSE: %.4f", val_losses[end]))
            @info msg
        end
    end

    return train_losses, val_losses
end

