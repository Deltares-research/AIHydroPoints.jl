# abstract_flux_model.jl
#
# Defines two types:
#
#   AbstractFluxModel <: AbstractModel   — abstract; implements predict,
#       save_params, load_params! once, in terms of get_flux_model and the
#       three customisation points preprocess / forward / postprocess!.
#
#   MyFluxModel <: AbstractFluxModel     — concrete; holds a Flux chain and a
#       settings dict.  Concrete domain models (TideModel, SurgeModel, …) are
#       constructed as MyFluxModel values, with domain-specific preprocess /
#       forward / postprocess! methods dispatching on the model type.

using Flux
using JLD2

# ──────────────────────────────────────────────────────────────────────────────
# AbstractFluxModel
# ──────────────────────────────────────────────────────────────────────────────

"""
    AbstractFluxModel <: AbstractModel

Abstract supertype for all Flux.jl-based AI-Hydro forecast models.

Sits between `AbstractModel` and concrete model types, implementing the generic
Flux machinery (`predict`, `save_params`, `load_params!`) once while leaving
data mapping and network architecture as customisation points.

```
AbstractModel
    └── AbstractFluxModel   — implements predict, save_params, load_params!
            └── MyFluxModel — imaginary concrete struct: Flux chain + settings dict;
                              domain models are MyFluxModel values with
                              domain-specific preprocess / forward / postprocess
```

## Tensor layout

Tensors passed between `preprocess`, `forward`, and `postprocess` use the
canonical column-major layout:

- **Input tensor** (output of `preprocess`):  `(locations, features, time_lag, time)`
- **Output tensor** (output of `forward`):    `(locations, features, time)`

`time` serves as the batch dimension so that the whole time-series is processed
in a single forward pass.

## Interface implemented at this level

`predict`, `save_params`, and `load_params!` are provided here and are inherited
by all subtypes without modification.

## Required customisation points

| Method | Signature | Purpose |
|---|---|---|
| `preprocess`    | `(m::M, input::Dict{String,TimeSeries}) -> (Array{Float32,4}, Dict{String,TimeSeries})` | Build input tensor and pre-allocate output |
| `forward`       | `(m::M, x::Array{Float32,4}) -> Array{Float32,3}` | Run Flux forward pass |
| `postprocess!`  | `(output::Dict{String,TimeSeries}, m::M, y::Array{Float32,3})` | Fill pre-allocated output in-place |
| `get_flux_model`| `(m::M) -> <Flux model>` | Return the underlying Flux model |
| `get_settings`  | `(m::M) -> Dict{String,Any}` | Return model settings |
"""
abstract type AbstractFluxModel <: AbstractModel end

# ──────────────────────────────────────────────────────────────────────────────
# predict — implemented once for all AbstractFluxModel subtypes
# ──────────────────────────────────────────────────────────────────────────────

"""
    predict(model::AbstractFluxModel, input::Dict{String, TimeSeries})
        -> Dict{String, TimeSeries}

Run inference by chaining `preprocess → forward → postprocess!`.

`preprocess` returns both the input tensor `(locations, features, time_lag, time)`
and a pre-allocated output `Dict{String, TimeSeries}` with the correct metadata
(times, station names, coordinates) and zero-initialised values.
`forward` runs the Flux model and returns `(locations, features, time)`.
`postprocess!` fills the pre-allocated output values in-place.
"""
function predict(model::AbstractFluxModel, input::Dict{String, TimeSeries})
    tensor, output = preprocess(model, input)
    y = forward(model, tensor)
    postprocess!(output, model, y)
    return output
end

# ──────────────────────────────────────────────────────────────────────────────
# save_params / load_params! — implemented once using get_flux_model
# ──────────────────────────────────────────────────────────────────────────────

"""
    save_params(model::AbstractFluxModel, file::String; overwrite::Bool=false)

Serialise trained Flux weights to `file` (JLD2 format) via `Flux.state`.
Settings are **not** included; retrieve them with `get_settings`.

Throws an error if the parent directory does not exist, or if the file already
exists and `overwrite` is `false`.
"""
function save_params(model::AbstractFluxModel, file::String; overwrite::Bool=false)
    isdir(dirname(file)) || error("directory does not exist: $(dirname(file))")
    !overwrite && isfile(file) && error("file already exists (use overwrite=true): $file")
    flux_model = get_flux_model(model)
    jldsave(file; model_state = Flux.state(flux_model))
end

"""
    load_params!(model::AbstractFluxModel, file::String)

Load trained weights from `file` into `model` in-place via `Flux.loadmodel!`.
Build the model from its settings first so the architecture matches the file.
"""
function load_params!(model::AbstractFluxModel, file::String)
    flux_model = get_flux_model(model)
    model_state = JLD2.load(file, "model_state")
    Flux.loadmodel!(flux_model, model_state)
    return model
end

# ──────────────────────────────────────────────────────────────────────────────
# Customisation-point fallbacks — clear errors for missing implementations
# ──────────────────────────────────────────────────────────────────────────────

"""
    preprocess(model::AbstractFluxModel, input::Dict{String, TimeSeries})
        -> (Array{Float32, 4}, Dict{String, TimeSeries})

Map `input` to an input tensor and a pre-allocated output container.

Returns a tuple `(tensor, output)` where:
- `tensor` has shape `(locations, features, time_lag, time)` and contains the
  scaled, lagged input data ready for the Flux forward pass.
- `output` is a `Dict{String, TimeSeries}` with the correct output metadata
  (variable names, station names, coordinates, time axis) and zero-initialised
  `values` matrices.  `postprocess!` will fill these in-place.

Responsibilities:
- Select and order input variables and locations.
- Apply input scaling or normalisation.
- Assemble the lagged input window (`time_lag = 1` means no lag).
- Allocate the output `TimeSeries` objects with `zeros(Float32, ...)` values.

Must be implemented for each concrete model type.
"""
function preprocess(model::AbstractFluxModel, input::Dict{String, TimeSeries})
    error("preprocess not implemented for $(typeof(model))")
end

"""
    forward(model::AbstractFluxModel, x::Array{Float32, 4})
        -> Array{Float32, 3}

Reshape `x` as required and run the Flux forward pass.
Returns `(locations, features, time)`.

Typical reshapes before calling the chain:
```julia
# Dense model
x_flat = reshape(x, locations * features * time_lag, time)

# 1-D temporal convolution
x_flat = reshape(x, locations * features, time_lag, time)
```

Must be implemented for each concrete model type.
"""
function forward(model::AbstractFluxModel, x::Array{Float32, 4})
    error("forward not implemented for $(typeof(model))")
end

"""
    postprocess!(output::Dict{String, TimeSeries}, model::AbstractFluxModel,
                 y::Array{Float32, 3})

Fill the pre-allocated `output` with values from the Flux output tensor `y` of
shape `(locations, features, time)`.

`output` is the dict returned by `preprocess`; its `TimeSeries` values matrices
already have the right shape and can be written to with `.=`.  Apply any inverse
scaling here before writing.

Must be implemented for each concrete model type.
"""
function postprocess!(output::Dict{String, TimeSeries}, model::AbstractFluxModel,
                      y::Array{Float32, 3})
    error("postprocess! not implemented for $(typeof(model))")
end

"""
    get_flux_model(model::AbstractFluxModel) -> <Flux model>

Return the underlying Flux chain stored in `model`.  Used by `save_params` and
`load_params!`.

`MyFluxModel` implements this automatically via its `flux_model` field.
Custom subtypes of `AbstractFluxModel` must implement it themselves.
"""
function get_flux_model(model::AbstractFluxModel)
    error("get_flux_model not implemented for $(typeof(model))")
end

# ──────────────────────────────────────────────────────────────────────────────
# FluxModel — concrete struct that domain models use
# ──────────────────────────────────────────────────────────────────────────────

"""
    MyFluxModel

Concrete subtype of `AbstractFluxModel` that wraps any Flux chain together with an inference-time settings dictionary.

Intended as the base concrete type for domain models (`TideModel`, `SurgeModel`,
…).  Domain-specific behaviour is provided by `preprocess`, `forward`, and
`postprocess` methods that dispatch on `MyFluxModel`.

```julia
# Construct
model = MyFluxModel(my_chain, settings)

# Domain-specific methods
preprocess(m::MyFluxModel, input) = ...
forward(m::MyFluxModel, x)        = ...
postprocess(m::MyFluxModel, y)    = ...
```

`get_flux_model` and `get_settings` are already implemented for `MyFluxModel`.
`predict`, `save_params`, and `load_params!` are inherited from
`AbstractFluxModel`.
"""
mutable struct MyFluxModel <: AbstractFluxModel
    flux_model
    settings   :: Dict{String, Any}
end

get_flux_model(m::MyFluxModel) = m.flux_model
get_settings(m::MyFluxModel)   = m.settings

# ──────────────────────────────────────────────────────────────────────────────
# write_outputs — default implementation for all AbstractFluxModel subtypes
# ──────────────────────────────────────────────────────────────────────────────

"""
    write_outputs(model::AbstractFluxModel, data::Dict, output_settings::Dict)

Generate outputs for all entries in `output_settings["outputs"]`.  `data` is
the dict returned by `load_data`.  Each entry selects a split (and optionally a
`timerange` sub-window) and controls which outputs to produce.

See `docs/output_settings.md` for the full schema and defaults.
"""
function write_outputs(model::AbstractFluxModel, data::Dict, output_settings::Dict)
    save_dir = get_settings(model)["model_dir"]
    outputs  = get(output_settings, "outputs",
                   [Dict{String,Any}("split" => "test")])

    for entry in outputs
        split = entry["split"]
        haskey(data, split) || continue

        name      = get(entry, "name",      split)
        timerange = get(entry, "timerange", nothing)

        do_timeseries = get(entry, "timeseries",   split == "testing")
        do_scatter    = get(entry, "scatter",      false)
        do_stats      = get(entry, "write_stats",  split == "testing")
        do_series     = get(entry, "write_series", false)

        if do_timeseries || do_scatter || do_stats || do_series
            out = predict(model, data[split].input)

            if do_timeseries
                subdir = joinpath(save_dir, "$(name)_timeseries")
                mkpath(subdir)
                _plot_station_series(out, data[split].target, subdir;
                                     timerange = timerange)
            end

            if do_scatter
                subdir = joinpath(save_dir, "$(name)_scatter")
                mkpath(subdir)
                _plot_station_scatter(out, data[split].target, subdir;
                                      timerange = timerange)
            end

            if do_stats
                _write_station_stats(out, data[split].target,
                                     joinpath(save_dir, "stats_$(name).csv");
                                     timerange = timerange)
            end

            if do_series
                fmt = get(output_settings, "series_format", "netcdf")
                _write_station_series(out, data[split].target, save_dir, name, fmt;
                                      timerange = timerange)
            end
        end
    end

    return nothing
end
