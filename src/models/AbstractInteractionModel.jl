# AbstractInteractionModel.jl
#
# Intermediate abstract type for all interaction models.  Sits between
# AbstractFluxModel and concrete models (ConvInteractionModel, etc.) and
# implements the shared interaction-specific logic.
#
# The interaction model predicts waterlevel from lagged tide and surge at each
# station.  Station identity is encoded via one-hot vectors so each
# (station × time) pair is an independent sample, exactly as in wave models.
#
# Input dict keys:  "tide", "surge"
# Target dict keys: "waterlevel"
# Output dict keys: "waterlevel"
#
# Normalization: Z-score, computed from training data and stored in settings
# under "input_mu", "input_std", "output_mu", "output_std".
#
# Concrete subtypes must implement: get_flux_model, get_settings, forward.

using Flux
using Printf: @sprintf
using ProgressMeter: Progress, next!
using Statistics: mean, std

# ──────────────────────────────────────────────────────────────────────────────
# Abstract type
# ──────────────────────────────────────────────────────────────────────────────

"""
    AbstractInteractionModel <: AbstractFluxModel

Abstract supertype for tide-surge interaction models.  Provides shared
implementations of `preprocess`, `postprocess!`, `train_model!`, and
`plot_series`.

## Required settings keys

| Key | Description |
|---|---|
| `"nlocations_output"` | Number of output (waterlevel) locations |
| `"nlags"`     | Number of lagged time steps used as input |

The following are populated automatically by `train_model!` on first call:
`"out_names"`, `"out_lons"`, `"out_lats"`, `"out_quantity"`, and the
Z-score normalisation statistics `"input_mu"`, `"input_std"`, `"output_mu"`,
`"output_std"`.

## Tensor layout

`preprocess` produces `(x_station, x_ts)` where:
- `x_station :: Bool (nstations, nstations * ntimes_valid)` — one-hot station encoding
- `x_ts :: Float32 (nlags, 2, nstations * ntimes_valid)` — Z-scored tide+surge lags,
  with channel 1 = surge and channel 2 = tide for each station.

Samples are ordered: for each valid time step (outer), then for each station (inner).
`forward` must accept this tuple and return `(nstations, 1, ntimes_valid)`.

## Input convention

`input` must contain:
- `"tide"`  — tidal signal `(nstations, ntimes)`
- `"surge"` — storm surge `(nstations, ntimes)`

`target` must contain:
- `"waterlevel"` — observed waterlevel `(nstations, ntimes)` (also carries station metadata)

## Concrete subtypes must implement

- `get_flux_model(m)` — return the underlying Flux model
- `get_settings(m)` — return `Dict{String, Any}`
- `forward(m, x::Tuple) -> Array{Float32, 3}`
"""
abstract type AbstractInteractionModel <: AbstractFluxModel end

# ──────────────────────────────────────────────────────────────────────────────
# Private helper: build raw (unnormalized) input blocks
# ──────────────────────────────────────────────────────────────────────────────

function _build_interaction_blocks(model::AbstractInteractionModel,
                                   input::Dict{String, TimeSeries})
    settings  = get_settings(model)
    nlags     = settings["nlags"]

    ts_tide  = input["tide"]
    ts_surge = input["surge"]

    tide_vals  = get_values(ts_tide)    # (nstations, ntimes)
    surge_vals = get_values(ts_surge)   # (nstations, ntimes)

    nstations    = size(tide_vals, 1)
    ntimes       = size(tide_vals, 2)
    ntimes_valid = ntimes - nlags + 1
    nsamples     = nstations * ntimes_valid

    # x_ts: (nlags, 2, nsamples)  — channel 1 = surge, channel 2 = tide
    x_ts = zeros(Float32, nlags, 2, nsamples)
    for (t_idx, itime) in enumerate(nlags:ntimes)
        for s in 1:nstations
            isample = (t_idx - 1) * nstations + s
            x_ts[:, 1, isample] .= surge_vals[s, itime-nlags+1:itime]
            x_ts[:, 2, isample] .= tide_vals[s,  itime-nlags+1:itime]
        end
    end

    # One-hot station encoding: (nstations, nstations * ntimes_valid)
    station_arr = collect(1:nstations) * ones(Int, ntimes_valid)'
    x_station   = Flux.onehotbatch(station_arr[:], 1:nstations)

    return x_station, x_ts
end

# ──────────────────────────────────────────────────────────────────────────────
# preprocess — shared across all interaction models
# ──────────────────────────────────────────────────────────────────────────────

"""
    preprocess(model::AbstractInteractionModel, input::Dict{String, TimeSeries})
        -> (Tuple, Dict{String, TimeSeries})

Build one-hot station and lagged tide+surge tensors from `input`, Z-score
the input blocks using statistics stored in the model settings (default: no
scaling if statistics are not yet set), and pre-allocate the output
`TimeSeries`.

Returns `((x_station, x_ts), output)` where:
- `x_station :: Bool (nstations, nstations * ntimes_valid)` — one-hot
- `x_ts :: Float32 (nlags, 2, nstations * ntimes_valid)` — Z-scored tide+surge blocks
- `output` is `Dict("waterlevel" => ts)` with `ts.values` zero-initialised.

Station metadata is read from `settings["out_names"]` etc. when available,
falling back to the metadata of `input["tide"]`.
"""
function preprocess(model::AbstractInteractionModel, input::Dict{String, TimeSeries})
    settings  = get_settings(model)
    nlags     = settings["nlags"]
    input_mu  = Float32(get(settings, "input_mu",  0.0))
    input_std = Float32(get(settings, "input_std", 1.0))

    if haskey(settings, "out_names")
        out_names = settings["out_names"]
        input = Dict(k => _check_and_align_locations(v, out_names, "input[\"$k\"]")
                     for (k, v) in input)
    end

    x_station, x_ts = _build_interaction_blocks(model, input)
    x_ts_norm = (x_ts .- input_mu) ./ input_std

    nstations    = size(x_station, 1)
    times_valid  = get_times(input["tide"])[nlags:end]
    ntimes_valid = length(times_valid)

    ts_ref = input["tide"]
    names = get(settings, "out_names", get_names(ts_ref))
    lons  = get(settings, "out_lons",  Float64.(get_longitudes(ts_ref)))
    lats  = get(settings, "out_lats",  Float64.(get_latitudes(ts_ref)))
    qty   = get(settings, "out_quantity", "waterlevel")

    out_ts = TimeSeries(
        zeros(Float32, nstations, ntimes_valid),
        times_valid, names, lons, lats, qty, string(typeof(model)),
    )
    out_key = first(get(settings, "out_quantities", ["waterlevel"]))
    return (x_station, x_ts_norm), Dict{String, TimeSeries}(out_key => out_ts)
end

# ──────────────────────────────────────────────────────────────────────────────
# postprocess! — shared across all interaction models
# ──────────────────────────────────────────────────────────────────────────────

"""
    postprocess!(output::Dict{String, TimeSeries}, model::AbstractInteractionModel,
                 y::Array{Float32, 3})

Write waterlevel predictions from `y` into `output["waterlevel"]` in-place,
applying the inverse Z-score transform.  `y` has shape `(nstations, 1, ntimes_valid)`.
"""
function postprocess!(output::Dict{String, TimeSeries}, model::AbstractInteractionModel,
                      y::Array{Float32, 3})
    settings   = get_settings(model)
    output_mu  = Float32(get(settings, "output_mu",  0.0))
    output_std = Float32(get(settings, "output_std", 1.0))
    out_key = first(keys(output))
    output[out_key].values .= y[:, 1, :] .* output_std .+ output_mu
end

# ──────────────────────────────────────────────────────────────────────────────
# train_model! — shared across all interaction models
# ──────────────────────────────────────────────────────────────────────────────

"""
    train_model!(model::AbstractInteractionModel, train_settings::TrainingSettings,
                 input::Dict{String, TimeSeries}, target::Dict{String, TimeSeries})
        -> (Vector{Float32}, Vector{Float32})

Train the model in-place using minibatch gradient descent (Adam) over
`(station × time)` samples.

`input` must contain `"tide"` and `"surge"`; `target` must contain
`"waterlevel"`.

On first call, `"out_names"`, `"out_lons"`, `"out_lats"`, `"out_quantity"`,
and the Z-score normalisation statistics are added to the model settings.

If `train_settings.validation_split > 0`, the last fraction of the time
series is held out for validation.

Returns `(train_losses, val_losses)` as `Vector{Float32}` per epoch;
`val_losses` is empty when `validation_split == 0`.
"""
function train_model!(model::AbstractInteractionModel, train_settings::TrainingSettings,
                      input::Dict{String, TimeSeries}, target::Dict{String, TimeSeries};
                      val_input::Union{Dict{String,TimeSeries},Nothing}  = nothing,
                      val_target::Union{Dict{String,TimeSeries},Nothing} = nothing)

    settings = get_settings(model)
    nlags    = settings["nlags"]

    # Populate output metadata on first call
    if !haskey(settings, "out_names")
        ts_ref = first(values(target))
        settings["out_names"]    = get_names(ts_ref)
        settings["out_lons"]     = Float64.(get_longitudes(ts_ref))
        settings["out_lats"]     = Float64.(get_latitudes(ts_ref))
        settings["out_quantity"] = get_quantity(ts_ref)
    end

    nstations = settings["nlocations_output"]

    # Build raw (unnormalized) input blocks
    x_station, x_ts = _build_interaction_blocks(model, input)
    ntimes_valid = size(x_ts, 3) ÷ nstations

    # Build raw target: (1, nstations * ntimes_valid)
    wl_vals = Float32.(get_values(first(values(target))))
    y_raw   = reshape(wl_vals[:, nlags:end], 1, :)

    # Temporal train/val split
    n_val_times   = round(Int, train_settings.validation_split * ntimes_valid)
    has_val       = n_val_times > 0
    n_train_times = ntimes_valid - n_val_times
    n_tr_samps    = n_train_times * nstations

    # Compute Z-score statistics from the training portion only
    x_ts_train = x_ts[:, :, 1:n_tr_samps]
    y_train_raw = y_raw[:, 1:n_tr_samps]

    input_mu   = Float32(mean(x_ts_train))
    input_std  = max(Float32(std(x_ts_train)), 1f-6)
    output_mu  = Float32(mean(y_train_raw))
    output_std = max(Float32(std(y_train_raw)), 1f-6)

    settings["input_mu"]   = input_mu
    settings["input_std"]  = input_std
    settings["output_mu"]  = output_mu
    settings["output_std"] = output_std

    # Normalize full dataset
    x_ts_norm = (x_ts .- input_mu) ./ input_std
    y_norm    = (y_raw .- output_mu) ./ output_std

    # Split into train / val
    x_s = x_station[:, 1:n_tr_samps]
    x_i = x_ts_norm[:, :, 1:n_tr_samps]
    y   = y_norm[:, 1:n_tr_samps]
    n_train = size(y, 2)

    x_s_val = x_i_val = y_val = nothing
    if has_val
        x_s_val = x_station[:, n_tr_samps+1:end]
        x_i_val = x_ts_norm[:, :, n_tr_samps+1:end]
        y_val   = y_norm[:, n_tr_samps+1:end]
    end

    flux_model = get_flux_model(model)
    opt_state  = Flux.setup(Adam(train_settings.learning_rate), flux_model)
    nbatch     = min(train_settings.nbatches, n_train)

    checkpoint_dir = get(settings, "model_dir", nothing)

    train_losses  = Float32[]
    val_losses    = Float32[]
    showvalues    = Pair{String, String}[]
    progress      = Progress(train_settings.nepochs; desc="Training: ", showspeed=true)
    log_every     = max(1, train_settings.nepochs ÷ 10)
    best_val_rmse = Inf32

    for epoch in 1:train_settings.nepochs
        idx = sortperm(rand(n_train))[1:nbatch]
        _, grads = Flux.withgradient(flux_model) do m
            Flux.mse(m((x_s[:, idx], x_i[:, :, idx])), y[:, idx])
        end
        Flux.update!(opt_state, flux_model, grads[1])

        train_rmse = sqrt(Flux.mse(flux_model((x_s, x_i)), y))
        push!(train_losses, train_rmse)

        empty!(showvalues)
        push!(showvalues, "train RMSE" => @sprintf("%.4f", train_rmse))
        if has_val
            val_rmse = sqrt(Flux.mse(flux_model((x_s_val, x_i_val)), y_val))
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

