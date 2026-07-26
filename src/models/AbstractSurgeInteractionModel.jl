# AbstractSurgeInteractionModel.jl
#
# Intermediate abstract type for surge-interaction models.  Extends
# AbstractSurgeModel: predicts the practical surge (water level minus tide) from the
# same wind-stress / pressure forcing as the surge models, PLUS a tide input at the
# output stations that drives a tide-surge interaction folded into the surge model.
#
# Input dict keys:  "stress_x"/"stress_y" (or "wind_x"/"wind_y"), "pressure", "tide"
# Target dict keys: "surge"   (= practical surge, provided directly)
# Output dict keys: "surge"
#
# Only the predict-form preprocess is specialised here (to add the tide lag
# windows); the train-form preprocess, postprocess!, forward and train_model! are
# all inherited from AbstractSurgeModel.
#
# Concrete subtypes must implement: get_flux_model, get_settings.

using Flux

"""
    AbstractSurgeInteractionModel <: AbstractSurgeModel

Abstract supertype for surge-interaction models.  These extend the surge models
with a `tide` input at the output stations and fold the tide-surge interaction
into the surge prediction.  All surge-model machinery is inherited
(`_surge_lag_windows`, the train-form `preprocess`, `postprocess!`, `forward`,
`train_model!`); the only addition is the tide lag windows in the predict-form
`preprocess`.

## Required settings keys

| Key | Description |
|---|---|
| `"nlocations_output"` | Number of output (surge / water-level) locations |
| `"nlocations_input"`  | Number of input (forcing) locations |
| `"nlags"`             | Number of lagged time steps used as input |

## Input / target convention

`input` must contain the surge forcing (`stress_x`/`stress_y` or `wind_x`/`wind_y`,
and `pressure`) at the `nlocations_input` input stations, and `tide` at the
`nlocations_output` output stations.  `target` must contain `"surge"` — the
practical surge (water level minus tide), provided directly.

## Concrete subtypes must implement

- `get_flux_model(m)` — return the underlying Flux model
- `get_settings(m)` — return `Dict{String, Any}`

The Flux model receives the tuple `(f_flat, t_lags)` and must return a 2-D
`(nlocations_output, ntimes_valid)` array.
"""
abstract type AbstractSurgeInteractionModel <: AbstractSurgeModel end

"""
    _tide_lag_windows(model::AbstractSurgeInteractionModel, input)
        -> t_lags :: (nlags, nlocations_output, nvalid)

Slice the `tide` input at the output stations into lag windows.  `t_lags[:, P, i]`
is the `nlags`-step tide history at output station `P` ending at valid step `i`,
matching the valid batch-time steps of the surge forcing windows.

The tide is aligned to the model's output stations (`out_names`) when available.
"""
function _tide_lag_windows(model::AbstractSurgeInteractionModel,
                           input::Dict{String, TimeSeries})
    settings  = get_settings(model)
    nstations = settings["nlocations_output"]
    nlags     = settings["nlags"]

    haskey(input, "tide") ||
        error("$(typeof(model)): input is missing the required \"tide\" time series.")

    tide_ts = input["tide"]
    if haskey(settings, "out_names")
        tide_ts = _check_and_align_locations(tide_ts, settings["out_names"], "input[\"tide\"]")
    end
    tide = Float32.(get_values(tide_ts))              # (nstations, ntimes)

    ntimes      = size(tide, 2)
    valid_range = nlags:ntimes
    nvalid      = length(valid_range)

    t_lags = zeros(Float32, nlags, nstations, nvalid)
    for (i, t) in enumerate(valid_range)
        @views t_lags[:, :, i] .= permutedims(tide[:, t-nlags+1:t], (2, 1))  # (nlags, nstations)
    end
    return t_lags
end

"""
    preprocess(model::AbstractSurgeInteractionModel, input)
        -> ((f_flat, t_lags), output)

Predict-form preprocess.  Builds the flat forcing vector `f_flat` (via the shared
`_surge_lag_windows`, exactly as `LinearSurgeModel`) and the tide lag windows
`t_lags` at the output stations, returning them as a 2-tuple plus the pre-allocated
`Dict("surge" => ts)` output.

- `f_flat :: (3*nlocations_input*nlags, nvalid)`
- `t_lags :: (nlags, nlocations_output, nvalid)`

The two blocks live on different station grids (input vs output), so they are kept
as separate tensors sharing only the last (batch-time) axis.  The train-form
`preprocess`, `postprocess!`, `forward` and `train_model!` are inherited from
`AbstractSurgeModel`.
"""
function preprocess(model::AbstractSurgeInteractionModel, input::Dict{String, TimeSeries})
    # Forcing lives on the input grid, tide on the output grid — extract them
    # separately.  `_surge_lag_windows` aligns its whole dict to `in_names`, so it
    # must not see `tide` (which is on the output stations, not the forcing grid).
    forcing = Dict{String, TimeSeries}(k => v for (k, v) in input if k != "tide")
    sx, sy, pr, times_valid = _surge_lag_windows(model, forcing)
    nvalid    = size(sx, 3)
    x_stacked = vcat(sx, sy, pr)                 # (3*nwind, nlags, nvalid)
    f_flat    = reshape(x_stacked, :, nvalid)    # (3*nwind*nlags, nvalid)

    t_lags = _tide_lag_windows(model, input)     # (nlags, nstations, nvalid)
    size(t_lags, 3) == nvalid || error(
        "$(typeof(model)): tide and forcing have mismatched valid lengths " *
        "($(size(t_lags, 3)) vs $nvalid) — check that tide and forcing share a time grid.")

    return (f_flat, t_lags), _alloc_surge_output(model, times_valid)
end
