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
function _aligned_surge_forcing(model::AbstractSurgeModel, input::Dict{String, TimeSeries})
    settings = get_settings(model)
    nwind    = settings["nlocations_input"]
    nlags    = settings["nlags"]

    if haskey(settings, "in_names")
        in_names = settings["in_names"]
        input = Dict(k => _check_and_align_locations(v, in_names, "input[\"$k\"]")
                     for (k, v) in input)
    end

    stress_x, stress_y = _get_stress(input)
    press = Float32.(2e-4 .* (get_values(input["pressure"]) .- 1e5))
    times       = get_times(input[_wind_key(input)])
    ntimes      = length(times)
    valid_range = nlags:ntimes
    return stress_x, stress_y, press, times, nwind, nlags, valid_range
end

# ──────────────────────────────────────────────────────────────────────────────
# Lazy batch input — raw forcing kept in memory, lags built per minibatch
# ──────────────────────────────────────────────────────────────────────────────

"""
    SurgeLagSource <: LazyBatchInput

Holds aligned wind-stress and scaled-pressure matrices plus the valid time indices.
Used by [`LinearSurgeModel`](@ref) preprocess so training never materialises the
full `(3*nlocations_input*nlags, ntimes_valid)` feature matrix (~32 GB at
317 stations × 20 years); batches are built on demand via [`materialize_batch`](@ref).
"""
struct SurgeLagSource <: LazyBatchInput
    stress_x     :: Matrix{Float32}
    stress_y     :: Matrix{Float32}
    press        :: Matrix{Float32}
    nwind        :: Int
    nlags        :: Int
    valid_range  :: UnitRange{Int}
end

nsamples(src::SurgeLagSource) = length(src.valid_range)

function subset(src::SurgeLagSource, sample_idx)
    return SurgeLagSource(src.stress_x, src.stress_y, src.press,
                          src.nwind, src.nlags, src.valid_range[sample_idx])
end

"""
    materialize_batch(src::SurgeLagSource, sample_cols) -> Matrix{Float32}

Build `(3*nwind*nlags, length(sample_cols))` for 1-based sample indices into
`valid_range`.  Layout matches [`_surge_lag_flat`](@ref).
"""
function materialize_batch(src::SurgeLagSource, sample_cols::AbstractVector{Int})
    n         = length(sample_cols)
    nfeatures = 3 * src.nwind * src.nlags
    x         = zeros(Float32, nfeatures, n)
    for (j, s) in enumerate(sample_cols)
        t = src.valid_range[s]
        col = view(x, :, j)
        offset = 0
        for mat in (src.stress_x, src.stress_y, src.press)
            for lag in 1:src.nlags
                col[offset + 1:offset + src.nwind] .= mat[:, t - src.nlags + lag]
                offset += src.nwind
            end
        end
    end
    return x
end

"""
    _surge_lag_source(model, input) -> (SurgeLagSource, times_valid)
"""
function _surge_lag_source(model::AbstractSurgeModel, input::Dict{String, TimeSeries})
    stress_x, stress_y, press, times, nwind, nlags, valid_range =
        _aligned_surge_forcing(model, input)
    src = SurgeLagSource(stress_x, stress_y, press, nwind, nlags, valid_range)
    return src, times[valid_range]
end

"""
    _surge_lag_flat(model::AbstractSurgeModel, input)
        -> (x_flat, times_valid)

Build the flat Dense input `(3*nlocations_input*nlags, ntimes_valid)` directly,
without materialising the intermediate `(nwind, nlags, nvalid)` lag-window
arrays.  Layout matches `vcat` + `reshape` in [`LinearSurgeModel`](@ref) preprocess.
Used by tests and callers that need the full matrix; training uses [`SurgeLagSource`](@ref).
"""
function _surge_lag_flat(model::AbstractSurgeModel, input::Dict{String, TimeSeries})
    src, times_valid = _surge_lag_source(model, input)
    return materialize_batch(src, collect(1:nsamples(src))), times_valid
end

function _surge_lag_windows(model::AbstractSurgeModel, input::Dict{String, TimeSeries})
    stress_x, stress_y, press, times, nwind, nlags, valid_range = _aligned_surge_forcing(model, input)
    nvalid = length(valid_range)

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
# train-form preprocess / postprocess! — shared across all surge models
# (forward and train_model! are inherited from AbstractFluxModel)
# ──────────────────────────────────────────────────────────────────────────────

"""
    preprocess(model::AbstractSurgeModel, input, target) -> (Tuple, Matrix)

Train-form preprocess: build the input tuple `x` (reusing the per-model 2-arg
predict form) and the lag-aligned target `y = get_values(target)[:, nlags:end]`
of shape `(nlocations_output, ntimes_valid)`.

The `nlags:end` trim drops the first `nlags-1` target columns that have no
complete lag window, matching the valid batch-time steps in `x`.  Consumed by the
generic `train_model!`.
"""
function preprocess(model::AbstractSurgeModel, input::Dict{String, TimeSeries},
                    target::Dict{String, TimeSeries})
    x, _  = preprocess(model, input)   # per-model 2-arg form builds the input tuple
    nlags = get_settings(model)["nlags"]
    y = Float32.(get_values(first(values(target))))[:, nlags:end]  # (nstations, nvalid)
    return x, y
end

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
