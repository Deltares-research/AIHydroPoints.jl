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

All surge models share the same *data extraction*: `wind_x`, `wind_y`, and
scaled pressure sliced into lag windows of shape
`(nlocations_input, nlags, ntimes_valid)` via [`_surge_lag_windows`](@ref).
Each concrete model then assembles those windows into whatever tensor layout its
layers need and returns them from `preprocess` as a **tuple** (a 1-tuple for the
single-input Dense/Conv models, a 2-tuple `(x_station, x_wind)` for the
attention model — see the per-model `preprocess` docstrings).

`forward` and `postprocess!` are provided generically at this level: `forward`
splats the tuple into the Flux model (`get_flux_model(m)(x...)`), which must
return a 2-D `(nlocations_output, ntimes_valid)` array. `train_model!` is a
single loop that works for both single- and multi-input models because
`Flux.DataLoader((x, y))` batches every tensor in the tuple along its shared
last (batch-time) axis.

## Concrete subtypes must implement

- `get_flux_model(m)` — return the underlying Flux model
- `get_settings(m)` — return `Dict{String, Any}`
- `preprocess(m, input) -> (Tuple, Dict{String,TimeSeries})` — per-model tensor assembly

The Flux model must return a 2-D `(nlocations_output, ntimes_valid)` array.
"""
abstract type AbstractSurgeModel <: AbstractFluxModel end

# ──────────────────────────────────────────────────────────────────────────────
# Shared data extraction — lag windows + output allocation
# ──────────────────────────────────────────────────────────────────────────────

"""
    _surge_lag_windows(model::AbstractSurgeModel, input::Dict{String, TimeSeries})
        -> (sx, sy, pr, times_valid)

Extract wind-stress and scaled-pressure forcing from `input` and slice it into
lag windows.  This is the part of preprocessing that is genuinely shared across
every surge model; each model then arranges these windows into its own tensor
layout.

Returns `(sx, sy, pr, times_valid)` where each of `sx`, `sy`, `pr` is a
`Float32` array of shape `(nlocations_input, nlags, ntimes_valid)`:

```
axis 1 → point (input location p),  varies fastest in memory
axis 2 → lag   (Δt, history step)
axis 3 → batch-time (valid step i)
```

For each valid batch-time step `i`, `sx[:, :, i]` holds the `nlags`-step history
ending at that step.  `times_valid` is the `Vector{DateTime}` of valid steps.

`ntimes_valid = ntimes - nlags + 1`.  Accepts either `"stress_x"`/`"stress_y"`
(used directly) or `"wind_x"`/`"wind_y"` (converted via `uv_to_stress_xy`);
pressure is scaled by `2e-4*(p - 1e5)`.
"""
function _surge_lag_windows(model::AbstractSurgeModel, input::Dict{String, TimeSeries})
    settings = get_settings(model)
    nwind    = settings["nlocations_input"]
    nlags    = settings["nlags"]

    # Align input locations to training-time order (errors on missing, drops extras)
    if haskey(settings, "in_names")
        in_names = settings["in_names"]
        input = Dict(k => _check_and_align_locations(v, in_names, "input[\"$k\"]")
                     for (k, v) in input)
    end

    stress_x, stress_y = _get_stress(input)                          # (nwind, ntimes)
    press = Float32.(2e-4 .* (get_values(input["pressure"]) .- 1e5)) # (nwind, ntimes)

    times       = get_times(input[_wind_key(input)])
    ntimes      = length(times)
    valid_range = nlags:ntimes
    nvalid      = length(valid_range)

    # Slice each forcing field into (point, lag, batch-time) = (nwind, nlags, nvalid).
    sx = zeros(Float32, nwind, nlags, nvalid)
    sy = zeros(Float32, nwind, nlags, nvalid)
    pr = zeros(Float32, nwind, nlags, nvalid)
    for (i, t) in enumerate(valid_range)
        sx[:, :, i] = stress_x[:, t-nlags+1:t]
        sy[:, :, i] = stress_y[:, t-nlags+1:t]
        pr[:, :, i] = press[   :, t-nlags+1:t]
    end
    return sx, sy, pr, times[valid_range]
end

"""
    _alloc_surge_output(model::AbstractSurgeModel, times_valid)
        -> Dict{String, TimeSeries}

Allocate the zero-initialised `Dict("surge" => ts)` output container for the
valid batch-time steps `times_valid`, reading station metadata (`out_names`,
`out_lons`, `out_lats`, `out_quantity`) from the model settings.

Requires `"out_names"`, `"out_lons"`, `"out_lats"` to be present in
`model.settings` — set automatically by `train_model!` on first use.
"""
function _alloc_surge_output(model::AbstractSurgeModel, times_valid)
    settings  = get_settings(model)
    nstations = settings["nlocations_output"]
    out_ts = TimeSeries(
        zeros(Float32, nstations, length(times_valid)),
        times_valid,
        settings["out_names"],
        settings["out_lons"],
        settings["out_lats"],
        get(settings, "out_quantity", "surge"),
        string(typeof(model)),
    )
    return Dict{String, TimeSeries}("surge" => out_ts)
end

# ──────────────────────────────────────────────────────────────────────────────
# forward / postprocess! — shared across all surge models
# ──────────────────────────────────────────────────────────────────────────────

"""
    forward(model::AbstractSurgeModel, x::Tuple) -> Array{Float32, 2}

Run the model's Flux network on the tuple `x` from `preprocess` and return a
2-D `(nlocations_output, ntimes)` array of surge predictions.

The tuple is splatted into the Flux model, so a 1-tuple `(x1,)` calls
`flux_model(x1)` and an N-tuple calls `flux_model(x1, …, xN)`.  Every surge Flux
model returns the 2-D `(nlocations_output, time)` shape directly.
"""
forward(model::AbstractSurgeModel, x::Tuple) = get_flux_model(model)(x...)

"""
    postprocess!(output::Dict{String, TimeSeries}, model::AbstractSurgeModel,
                 y::AbstractMatrix)

Write the 2-D surge predictions `y` of shape `(nlocations_output, ntimes)` into
`output["surge"]` in-place.
"""
function postprocess!(output::Dict{String, TimeSeries}, model::AbstractSurgeModel,
                      y::AbstractMatrix)
    output["surge"].values .= y
end

# ──────────────────────────────────────────────────────────────────────────────
# train_model! — shared across all surge models
# ──────────────────────────────────────────────────────────────────────────────

"""
    _take_last_dim(x::Tuple, idx) -> Tuple

Slice every tensor in `x` along its last (batch-time) axis at indices `idx`,
returning a new tuple of materialised arrays.  Used to split a preprocessed
input tuple into train/validation portions.
"""
_take_last_dim(x::Tuple, idx) = map(a -> copy(selectdim(a, ndims(a), idx)), x)

"""
    train_model!(model::AbstractSurgeModel, train_settings::TrainingSettings,
                 input::Dict{String, TimeSeries}, target::Dict{String, TimeSeries})
        -> (Vector{Float32}, Vector{Float32})

Train the model in-place using minibatch gradient descent (Adam).

This single loop serves every surge model — single-input (`LinearSurgeModel`,
`ConvSurgeModel`) and multi-input (`AttentionSurgeModel`) — because `preprocess`
always returns the input as a tuple `x`, `Flux.DataLoader((x, y))` batches every
tensor in that tuple along its shared last (batch-time) axis, and the Flux model
is called by splatting the batched tuple (`m(xb...)`).

`input` must contain `"wind_x"`, `"wind_y"`, and `"pressure"` (or the
`"stress_*"` equivalents). `target` must contain one variable (the surge ground
truth); its columns `nlags:end` correspond to the valid batch-time steps.

On first call, `"out_names"`, `"out_lons"`, `"out_lats"`, and `"out_quantity"`
are added to the model settings from the first `TimeSeries` in `target`.

If `val_input` / `val_target` are supplied they are used directly and
`validation_split` is ignored; otherwise the last `validation_split` fraction of
the time axis is held out. Returns `(train_losses, val_losses)` per epoch;
`val_losses` is empty when there is no validation data.
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

    nlags = settings["nlags"]

    # Build input tuple + target matrix (batch-time is the last axis of each).
    x_full, _ = preprocess(model, input)                              # Tuple
    y_full = Float32.(get_values(first(values(target))))[:, nlags:end]  # (nstations, nvalid)

    # Validation data: explicit split takes priority over validation_split
    if !isnothing(val_input)
        x_val, _ = preprocess(model, val_input)
        y_val    = Float32.(get_values(first(values(val_target))))[:, nlags:end]
        x, y     = x_full, y_full
        has_val  = true
    else
        nfull   = size(y_full, 2)
        n_val   = round(Int, train_settings.validation_split * nfull)
        has_val = n_val > 0
        if has_val
            n_train = nfull - n_val
            x       = _take_last_dim(x_full, 1:n_train)
            y       = y_full[:, 1:n_train]
            x_val   = _take_last_dim(x_full, n_train+1:nfull)
            y_val   = y_full[:, n_train+1:end]
        else
            x, y  = x_full, y_full
            x_val = y_val = nothing
        end
    end

    # Training loop
    flux_model = get_flux_model(model)
    opt_state  = Flux.setup(Adam(train_settings.learning_rate), flux_model)
    current_lr = Float64(train_settings.learning_rate)
    # DataLoader batches the nested tuple ((x1,…,xN), y) element-wise along the
    # last axis, yielding (xb::Tuple, yb) each iteration.
    loader = Flux.DataLoader((x, y); batchsize=train_settings.nbatches, shuffle=true)

    checkpoint_dir = get(settings, "model_dir", nothing)

    train_losses  = Float32[]
    val_losses    = Float32[]
    showvalues    = Pair{String,String}[]
    progress      = Progress(train_settings.nepochs; desc="Training: ", showspeed=true)
    log_every     = max(1, train_settings.nepochs ÷ 10)
    best_val_rmse = Inf32

    for epoch in 1:train_settings.nepochs
        for (xb, yb) in loader
            _, grads = Flux.withgradient(flux_model) do m
                Flux.mse(m(xb...), yb)
            end
            Flux.update!(opt_state, flux_model, grads[1])
        end

        train_rmse = sqrt(Flux.mse(flux_model(x...), y))
        push!(train_losses, train_rmse)

        empty!(showvalues)
        push!(showvalues, "train RMSE" => @sprintf("%.4f", train_rmse))
        if has_val
            val_rmse = sqrt(Flux.mse(flux_model(x_val...), y_val))
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

        if !isnothing(train_settings.lr_decay_factor) &&
                !isnothing(train_settings.lr_decay_rate) &&
                epoch % train_settings.lr_decay_rate == 0
            current_lr *= train_settings.lr_decay_factor
            Flux.Optimisers.adjust!(opt_state, current_lr)
        end
    end

    return train_losses, val_losses
end

