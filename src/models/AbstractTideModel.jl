# AbstractTideModel.jl
#
# Intermediate abstract type for all tide models.  Sits between AbstractFluxModel
# and concrete tide models (DeepONetTideModel, etc.) and implements shared
# tide-specific logic: preprocess, postprocess!, and train_model!.
#
# Concrete subtypes must implement: get_flux_model, get_settings, forward.
#
# Unlike surge models, tide models have no external forcing — inputs are computed
# from station lat/lon and time (Doodson numbers).  The input dict and target dict
# both carry "waterlevel" as the sole key.

using Flux
using Printf: @sprintf
using ProgressMeter: Progress, next!

"""
    AbstractTideModel <: AbstractFluxModel

Abstract supertype for tide models.  Provides shared implementations of
`preprocess`, `postprocess!`, and `train_model!` for models that predict
tides from station coordinates and astronomical time (Doodson numbers).

## Required settings keys

| Key | Description |
|---|---|
| `"freqs"` | Vector of tidal constituent names, e.g. `["M2","S2","K1",...]` |
| `"model_pars"` | Dict of architecture hyperparameters (subtype-specific) |

The following are populated automatically by `train_model!` on first call:
`"out_names"`, `"out_lons"`, `"out_lats"`, `"out_quantity"`.

## Tensor layout

`preprocess` produces `(x_station, x_doodson)` where:
- `x_station :: Float32 (4, nstations, ntimes)` — cos/sin lat and cos/sin lon per station
- `x_doodson :: Float32 (2*nfreqs, ntimes)` — cos/sin Doodson arguments per time step

`forward` must accept this tuple and return `(nstations, 1, ntimes)`.

## Input convention

Both `input` and `target` dicts carry a `"waterlevel"` key.
`preprocess` reads times and station metadata from `input["waterlevel"]`.
`train_model!` reads ground truth values from `target["waterlevel"]`.

## Concrete subtypes must implement

- `get_flux_model(m)` — return the underlying Flux model
- `get_settings(m)` — return `Dict{String, Any}`
- `forward(m, x::Tuple) -> Array{Float32, 3}`
"""
abstract type AbstractTideModel <: AbstractFluxModel end

# ──────────────────────────────────────────────────────────────────────────────
# preprocess — shared across all tide models
# ──────────────────────────────────────────────────────────────────────────────

"""
    preprocess(model::AbstractTideModel, input::Dict{String, TimeSeries})
        -> (Tuple, Dict{String, TimeSeries})

Build station-encoding and Doodson-argument tensors from `input["waterlevel"]`,
and pre-allocate the output `TimeSeries`.

Returns `((x_station, x_doodson), output)` where:
- `x_station :: Float32 (4, nstations, ntimes)` — `[cos(lat), sin(lat), cos(lon), sin(lon)]`
- `x_doodson :: Float32 (2*nfreqs, ntimes)` — `[cos(doodson); sin(doodson)]`
- `output` is `Dict("waterlevel" => ts)` with `ts.values` zero-initialised.

Tidal frequencies are taken from `model.settings["freqs"]`.
"""
function preprocess(model::AbstractTideModel, input::Dict{String, TimeSeries})
    settings  = get_settings(model)
    ts        = input["waterlevel"]
    times     = get_times(ts)
    lats      = get_latitudes(ts)
    lons      = get_longitudes(ts)
    names     = get_names(ts)
    nstations = length(lats)
    ntimes    = length(times)
    freqs     = settings["freqs"]
    nfreqs    = length(freqs)

    x_station = zeros(Float32, 4, nstations, ntimes)
    x_station[1, :, :] .= Float32.(cos.(deg2rad.(lats)))
    x_station[2, :, :] .= Float32.(sin.(deg2rad.(lats)))
    x_station[3, :, :] .= Float32.(cos.(deg2rad.(lons)))
    x_station[4, :, :] .= Float32.(sin.(deg2rad.(lons)))

    frequencies = primary_frequencies_as_doodson(freqs)
    doodson     = (get_doodson_eqvals(times) * frequencies)'   # (nfreqs, ntimes)
    x_doodson   = Float32.(vcat(cos.(doodson), sin.(doodson))) # (2*nfreqs, ntimes)

    out_ts = TimeSeries(
        zeros(Float32, nstations, ntimes),
        times,
        names,
        Float64.(lons),
        Float64.(lats),
        get_quantity(ts),
        string(typeof(model)),
    )
    output = Dict{String, TimeSeries}("waterlevel" => out_ts)

    return (x_station, x_doodson), output
end

# ──────────────────────────────────────────────────────────────────────────────
# postprocess! — shared across all tide models
# ──────────────────────────────────────────────────────────────────────────────

"""
    postprocess!(output::Dict{String, TimeSeries}, model::AbstractTideModel,
                 y::Array{Float32, 3})

Write tide predictions from `y` into `output["waterlevel"]` in-place.
`y` has shape `(nstations, 1, ntimes)`; the singleton feature dimension is dropped.
"""
function postprocess!(output::Dict{String, TimeSeries}, model::AbstractTideModel,
                      y::Array{Float32, 3})
    output["waterlevel"].values .= y[:, 1, :]
end

# ──────────────────────────────────────────────────────────────────────────────
# train_model! — shared across all tide models
# ──────────────────────────────────────────────────────────────────────────────

"""
    train_model!(model::AbstractTideModel, train_settings::TrainingSettings,
                 input::Dict{String, TimeSeries}, target::Dict{String, TimeSeries})
        -> (Vector{Float32}, Vector{Float32})

Train the model in-place using minibatch gradient descent (Adam), batching
over the time axis.

Both `input` and `target` must contain a `"waterlevel"` key.
`preprocess` uses `input["waterlevel"]` for times and station metadata.
Ground truth values are taken from `target["waterlevel"]`.

On first call, `"out_names"`, `"out_lons"`, `"out_lats"`, and `"out_quantity"`
are added to model settings from `target["waterlevel"]`.

Returns `(train_losses, val_losses)` as `Vector{Float32}` per epoch.
`val_losses` is empty when `validation_split == 0` and no explicit validation
data is provided.

If `val_input` / `val_target` are supplied they are used directly and
`validation_split` is ignored.
"""
function train_model!(model::AbstractTideModel, train_settings::TrainingSettings,
                      input::Dict{String, TimeSeries}, target::Dict{String, TimeSeries};
                      val_input::Union{Dict{String,TimeSeries},Nothing}  = nothing,
                      val_target::Union{Dict{String,TimeSeries},Nothing} = nothing)

    settings = get_settings(model)

    # Populate output metadata from target if not yet present
    if !haskey(settings, "out_names")
        ts_ref = target["waterlevel"]
        settings["out_names"]    = get_names(ts_ref)
        settings["out_lons"]     = Float64.(get_longitudes(ts_ref))
        settings["out_lats"]     = Float64.(get_latitudes(ts_ref))
        settings["out_quantity"] = get_quantity(ts_ref)
    end

    (x_station, x_doodson), _ = preprocess(model, input)
    ntimes = size(x_doodson, 2)

    y_all = Float32.(get_values(target["waterlevel"]))   # (nstations, ntimes)

    # Validation data: explicit split takes priority over validation_split
    if !isnothing(val_input)
        (x_st_val, x_w_val), _ = preprocess(model, val_input)
        y_val   = Float32.(get_values(val_target["waterlevel"]))
        has_val = true
        n_train = ntimes
        x_st    = x_station
        x_w     = x_doodson
        y       = y_all
    else
        n_val   = round(Int, train_settings.validation_split * ntimes)
        has_val = n_val > 0
        n_train = ntimes - n_val
        x_st      = x_station[:, :, 1:n_train]
        x_w       = x_doodson[:, 1:n_train]
        y         = y_all[:, 1:n_train]
        x_st_val  = has_val ? x_station[:, :, n_train+1:end] : nothing
        x_w_val   = has_val ? x_doodson[:, n_train+1:end]    : nothing
        y_val     = has_val ? y_all[:, n_train+1:end]         : nothing
    end

    flux_model = get_flux_model(model)
    opt_state  = Flux.setup(Adam(train_settings.learning_rate), flux_model)
    current_lr = Float64(train_settings.learning_rate)
    loader = Flux.DataLoader((x_st, x_w, y); batchsize=train_settings.nbatches, shuffle=true)

    checkpoint_dir = get(settings, "model_dir", nothing)

    train_losses  = Float32[]
    val_losses    = Float32[]
    showvalues    = Pair{String,String}[]
    progress      = Progress(train_settings.nepochs; desc="Training: ", showspeed=true)
    log_every     = max(1, train_settings.nepochs ÷ 10)
    best_val_rmse = Inf32

    for epoch in 1:train_settings.nepochs
        for (x_st_b, x_w_b, yb) in loader
            _, grads = Flux.withgradient(flux_model) do m
                Flux.mse(m(x_st_b, x_w_b), yb)
            end
            Flux.update!(opt_state, flux_model, grads[1])
        end

        train_rmse = sqrt(Flux.mse(flux_model(x_st, x_w), y))
        push!(train_losses, train_rmse)

        empty!(showvalues)
        push!(showvalues, "train RMSE" => @sprintf("%.4f", train_rmse))
        if has_val
            val_rmse = sqrt(Flux.mse(flux_model(x_st_val, x_w_val), y_val))
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

