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

"""
    preprocess(model::AbstractWaveModel, input, target) -> (Tuple, Matrix)

Train-form preprocess: build the input tuple `(x_station, x_input)` (reusing the
2-arg predict form) and the scaled target `y :: (1, nstations*ntimes_valid)`,
then **NaN-filter** input and target jointly — samples where any input block or
the target value is `NaN` are dropped from all three.

Consumed by the generic `train_model!`.  When explicit `val_input`/`val_target`
are supplied to `train_model!`, this runs on the validation set too (so the val
data is NaN-filtered as well).
"""
function preprocess(model::AbstractWaveModel, input::Dict{String, TimeSeries},
                    target::Dict{String, TimeSeries})
    settings   = get_settings(model)
    nlags      = settings["nlags"]
    wave_scale = Float32(get(settings, "wave_scale", 3.0))

    (x_station, x_input), _ = preprocess(model, input)   # reuse 2-arg for x
    y_flat = reshape(
        Float32.(get_values(target["wave_height"]))[:, nlags:end] ./ wave_scale,
        1, :,
    )

    # Drop (station × time) samples with any NaN in the input block or the target.
    nsamples = size(x_input, 3)
    valid = [i for i in 1:nsamples
             if !any(isnan, x_input[:, :, i]) && !isnan(y_flat[1, i])]
    return (x_station[:, valid], x_input[:, :, valid]), y_flat[:, valid]
end

# ──────────────────────────────────────────────────────────────────────────────
# postprocess! — shared across all wave models
# (forward and train_model! are inherited from AbstractFluxModel)
# ──────────────────────────────────────────────────────────────────────────────

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
