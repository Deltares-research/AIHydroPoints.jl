    # AbstractWaveModel.jl
#
# Intermediate abstract type for all wave models.  Sits between AbstractFluxModel
# and concrete wave models (ConvWaveModel, etc.) and implements shared wave-specific
# logic: preprocess, postprocess!, and train_model!.
#
# Unlike surge models, wave models encode station identity via one-hot vectors, so
# each (station × time) pair is an independent sample.  The model maps
#
#   (x_station, x_input) → wave_height prediction
#
# where x_station is one-hot and x_input is a lagged wind-stress block.
#
# Concrete subtypes must implement: get_flux_model, get_settings.
# (forward, postprocess!, and train_model! are shared here.)

using Flux
using Printf: @sprintf
using ProgressMeter: Progress, next!

# ──────────────────────────────────────────────────────────────────────────────
# Internal helper: convert (speed, direction) to scaled wind stress components
# ──────────────────────────────────────────────────────────────────────────────

function _wave_wind_to_stress(u10_values, udir_values, wind_scale)
    wind_x = Float32.(u10_values .* -sind.(udir_values) ./ wind_scale)
    wind_y = Float32.(u10_values .* -cosd.(udir_values) ./ wind_scale)
    for i in eachindex(wind_x)
        wind_x[i], wind_y[i] = uv_to_stress_xy(wind_x[i], wind_y[i])
    end
    return wind_x, wind_y
end

# ──────────────────────────────────────────────────────────────────────────────
# Abstract type
# ──────────────────────────────────────────────────────────────────────────────

"""
    AbstractWaveModel <: AbstractFluxModel

Abstract supertype for wave models.  Provides shared implementations of
`preprocess`, `postprocess!`, and `train_model!` for models that predict
wave height from lagged wind speed and direction at `nwind` locations.

## Required settings keys

| Key | Description |
|---|---|
| `"nlocations_output"` | Number of output (wave height) locations |
| `"nlocations_input"`  | Number of input (wind) locations |
| `"nlags"`             | Number of lagged time steps used as input |

Optional settings keys (with defaults):

| Key | Default | Description |
|---|---|---|
| `"wind_scale"` | `0.5` | Divisor applied to wind stress before input |
| `"wave_scale"` | `3.0` | Divisor applied to wave height targets during training; multiplied back in postprocess! |

The following are populated automatically by `train_model!` on first call:
`"out_names"`, `"out_lons"`, `"out_lats"`, `"out_quantity"`.

## Tensor layout

`preprocess` produces `(x_station, x_input)` where:
- `x_station :: Bool  (nlocations_output, nlocations_output * ntimes_valid)` — one-hot station encoding
- `x_input  :: Float32 (nlags, 2*nlocations_input, nlocations_output * ntimes_valid)` — lagged wind-stress blocks

Samples are ordered: for each valid time step (outer), then for each station (inner).
`forward` is provided generically (`get_flux_model(m)(x)`) and returns a 2-D
`(1, nlocations_output*ntimes_valid)` array — one value per `(station, time)`
sample; `postprocess!` reshapes it to `(nlocations_output, ntimes_valid)`.

## Input convention

`input` and `target` must both contain:
- `"wind_speed"` — wind speed in m/s `(nlocations_input, ntimes)`
- `"wind_direction"` — meteorological direction in degrees `(nlocations_input, ntimes)`
- `"wave_height"` — significant wave height in m `(nlocations_output, ntimes)` (also carries station metadata)

## Concrete subtypes must implement

- `get_flux_model(m)` — return the underlying Flux model
- `get_settings(m)` — return `Dict{String, Any}`

`forward`, `postprocess!`, and `train_model!` are shared; the Flux model must
accept the `(x_station, x_input)` tuple as a single argument and return a 2-D
`(1, nlocations_output*ntimes_valid)` array.
"""
abstract type AbstractWaveModel <: AbstractFluxModel end

# ──────────────────────────────────────────────────────────────────────────────
# preprocess — shared across all wave models
# ──────────────────────────────────────────────────────────────────────────────

"""
    preprocess(model::AbstractWaveModel, input::Dict{String, TimeSeries})
        -> (Tuple, Dict{String, TimeSeries})

Build one-hot station and lagged wind-stress tensors from `input`, and
pre-allocate the output `TimeSeries`.

Returns `((x_station, x_input), output)` where:
- `x_station :: Bool  (nlocations_output, nlocations_output * ntimes_valid)` — one-hot
- `x_input  :: Float32 (nlags, 2*nlocations_input, nlocations_output * ntimes_valid)` — wind stress blocks
- `output` is `Dict("wave_height" => ts)` with `ts.values` zero-initialised.

`ntimes_valid = ntimes - nlags + 1`. Samples are ordered: for each valid time
step (outer loop), then for each station (inner loop).

Wind speed and direction are converted to stress via `_wave_wind_to_stress`.
"""
function preprocess(model::AbstractWaveModel, input::Dict{String, TimeSeries})
    settings   = get_settings(model)
    nlags      = settings["nlags"]
    wind_scale = Float32(get(settings, "wind_scale", 0.5))

    if haskey(settings, "in_names")
        in_names = settings["in_names"]
        input = Dict(k => _check_and_align_locations(v, in_names, "input[\"$k\"]")
                     for (k, v) in input)
    end

    u10  = input["wind_speed"]
    udir = input["wind_direction"]

    times        = get_times(u10)
    ntimes       = length(times)
    ntimes_valid = ntimes - nlags + 1
    nstations    = settings["nlocations_output"]
    nwind        = size(get_values(u10), 1)

    wind_x, wind_y = _wave_wind_to_stress(get_values(u10), get_values(udir), wind_scale)

    # Lagged wind-stress blocks: (nlags, 2*nwind, nstations * ntimes_valid)
    # Ordering: for each t in nlags:ntimes (outer), for each station s (inner)
    x_input = zeros(Float32, nlags, 2 * nwind, nstations * ntimes_valid)
    for (t_idx, itime) in enumerate(nlags:ntimes)
        x_block = Float32.(vcat(wind_x[:, itime-nlags+1:itime],
                                wind_y[:, itime-nlags+1:itime]))'
        for s in 1:nstations
            x_input[:, :, (t_idx - 1) * nstations + s] .= x_block
        end
    end

    # One-hot station encoding: (nstations, nstations * ntimes_valid)
    station_arr = collect(1:nstations) * ones(Int, ntimes_valid)'   # (nstations, ntimes_valid)
    x_station   = Flux.onehotbatch(station_arr[:], 1:nstations)     # column-major: (t, s) ordering ✓

    # Pre-allocate output TimeSeries
    times_valid = times[nlags:end]
    names = settings["out_names"]
    lons  = settings["out_lons"]
    lats  = settings["out_lats"]
    qty   = get(settings, "out_quantity", "wave_height")
    out_ts = TimeSeries(
        zeros(Float32, nstations, ntimes_valid),
        times_valid, names, lons, lats, qty, string(typeof(model)),
    )
    return (x_station, x_input), Dict{String, TimeSeries}("wave_height" => out_ts)
end

# ──────────────────────────────────────────────────────────────────────────────
# forward / postprocess! — shared across all wave models
# ──────────────────────────────────────────────────────────────────────────────

"""
    forward(model::AbstractWaveModel, x::Tuple) -> Array{Float32, 2}

Run the model's Flux network on `(x_station, x_input)` from `preprocess`.  The
tuple is passed as a **single argument** (`get_flux_model(m)(x)`, not splatted —
the wave flux models unpack the tuple in their first layer).  Returns a 2-D
`(1, nstations*ntimes_valid)` array: one scaled wave-height value per
`(station, time)` sample.  `postprocess!` reshapes and unscales it.
"""
forward(model::AbstractWaveModel, x::Tuple) = get_flux_model(model)(x)

"""
    postprocess!(output::Dict{String, TimeSeries}, model::AbstractWaveModel,
                 y::AbstractMatrix)

Write wave-height predictions from the 2-D `y` of shape
`(1, nstations*ntimes_valid)` into `output["wave_height"]` in-place: reshape the
`(station × time)` samples back to `(nstations, ntimes_valid)` (station-fastest
within time, matching `preprocess`) and apply the inverse training scale
(`wave_scale`).
"""
function postprocess!(output::Dict{String, TimeSeries}, model::AbstractWaveModel,
                      y::AbstractMatrix)
    wave_scale        = Float32(get(get_settings(model), "wave_scale", 3.0))
    vals              = output["wave_height"].values        # (nstations, ntimes_valid)
    nstations, ntimes = size(vals)
    vals .= reshape(y, nstations, ntimes) .* wave_scale
end

# ──────────────────────────────────────────────────────────────────────────────
# train_model! — shared across all wave models
# ──────────────────────────────────────────────────────────────────────────────

"""
    train_model!(model::AbstractWaveModel, train_settings::TrainingSettings,
                 input::Dict{String, TimeSeries}, target::Dict{String, TimeSeries})
        -> (Vector{Float32}, Vector{Float32})

Train the model in-place using minibatch gradient descent (Adam) over
`(station × time)` samples.

`input` and `target` must contain `"wind_speed"`, `"wind_direction"`, and
`"wave_height"`.  Records where any input or target value is NaN are silently
dropped.

On first call, `"out_names"`, `"out_lons"`, `"out_lats"`, and `"out_quantity"`
are added to the model settings from the data.

If `train_settings.validation_split > 0`, the last fraction of the time series
(in the time dimension) is held out and its RMSE is shown in the progress bar.

Returns `(train_losses, val_losses)` as `Vector{Float32}` per epoch.
`val_losses` is empty when `validation_split == 0`.
"""
function train_model!(model::AbstractWaveModel, train_settings::TrainingSettings,
                      input::Dict{String, TimeSeries}, target::Dict{String, TimeSeries};
                      val_input::Union{Dict{String,TimeSeries},Nothing}  = nothing,
                      val_target::Union{Dict{String,TimeSeries},Nothing} = nothing)

    settings   = get_settings(model)
    wave_scale = Float32(get(settings, "wave_scale", 3.0))
    nlags      = settings["nlags"]

    # Populate output metadata from target on first call
    if !haskey(settings, "out_names")
        ts_ref = target["wave_height"]
        settings["out_names"]    = get_names(ts_ref)
        settings["out_lons"]     = Float64.(get_longitudes(ts_ref))
        settings["out_lats"]     = Float64.(get_latitudes(ts_ref))
        settings["out_quantity"] = get_quantity(ts_ref)
    end

    # Build full tiled tensors
    (x_station, x_input), _ = preprocess(model, input)
    nstations    = size(x_station, 1)
    nsamples     = size(x_input, 3)
    ntimes_valid = nsamples ÷ nstations

    # Target: (1, nstations * ntimes_valid), scaled
    y_flat = reshape(
        Float32.(get_values(target["wave_height"]))[:, nlags:end] ./ wave_scale,
        1, :,
    )

    # Temporal train/val split (split on time axis, then NaN-filter each part)
    n_val_times   = round(Int, train_settings.validation_split * ntimes_valid)
    has_val       = n_val_times > 0
    n_train_times = ntimes_valid - n_val_times
    n_tr_samps    = n_train_times * nstations
    n_val_samps   = n_val_times  * nstations

    function _nanfilter(xs, xi, yf, n)
        valid = [i for i in 1:n
                 if !any(isnan, xi[:, :, i]) && !isnan(yf[1, i])]
        return xs[:, valid], xi[:, :, valid], yf[:, valid]
    end

    x_s, x_i, y = _nanfilter(x_station, x_input, y_flat, n_tr_samps)

    x_s_val = x_i_val = y_val = nothing
    if has_val
        x_s_val, x_i_val, y_val = _nanfilter(
            x_station[:, n_tr_samps+1:end],
            x_input[:, :, n_tr_samps+1:end],
            y_flat[:, n_tr_samps+1:end],
            n_val_samps,
        )
    end

    # Bundle the two input tensors into a tuple so the loop matches the shared
    # surge/tide form: DataLoader((x, y)) yields (xb::Tuple, yb). Wave flux models
    # take the tuple as a SINGLE argument (`m(xb)`), unlike surge/tide (`m(xb...)`)
    # — that call-form difference is unified by `apply_flux` at the 20h dedup step.
    # NB: unlike surge/tide, `val_input`/`val_target` are ignored — wave validates
    #     only via the `validation_split` fraction (with the NaN-filter). Loop is
    #     otherwise identical to AbstractSurgeModel.train_model!.
    x     = (x_s, x_i)
    x_val = has_val ? (x_s_val, x_i_val) : nothing

    flux_model = get_flux_model(model)
    opt_state  = Flux.setup(Adam(train_settings.learning_rate), flux_model)
    current_lr = Float64(train_settings.learning_rate)
    loader = Flux.DataLoader((x, y); batchsize=train_settings.nbatches, shuffle=true)

    checkpoint_dir = get(get_settings(model), "model_dir", nothing)

    train_losses  = Float32[]
    val_losses    = Float32[]
    showvalues    = Pair{String, String}[]
    progress      = Progress(train_settings.nepochs; desc="Training: ", showspeed=true)
    log_every     = max(1, train_settings.nepochs ÷ 10)
    best_val_rmse = Inf32

    for epoch in 1:train_settings.nepochs
        for (xb, yb) in loader
            _, grads = Flux.withgradient(flux_model) do m
                Flux.mse(m(xb), yb)
            end
            Flux.update!(opt_state, flux_model, grads[1])
        end

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

        if !isnothing(train_settings.lr_decay_factor) &&
                !isnothing(train_settings.lr_decay_rate) &&
                epoch % train_settings.lr_decay_rate == 0
            current_lr *= train_settings.lr_decay_factor
            Flux.Optimisers.adjust!(opt_state, current_lr)
        end
    end

    return train_losses, val_losses
end

