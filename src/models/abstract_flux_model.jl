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
using Statistics: mean
using Printf: @sprintf
using ProgressMeter: Progress, next!
using hatyan_core: constituent_list

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

Each model owns its own tensor layout. The only contract enforced at this
level is the *shape of the containers* exchanged between the customisation
points, not the internal axis arrangement:

- `preprocess` returns `(x, output)` where `x` is a **tuple of tensors** — a
  1-tuple for single-input models (`LinearSurgeModel`, `ConvSurgeModel`), an
  N-tuple for multi-input models (`AttentionSurgeModel`, wave, interaction).
  Each model arranges its tensor(s) in the layout its Flux layers actually
  need; there is no shared "canonical" input layout.
- `forward` returns a 2-D array `(locations, time)` — predictions per location
  per batch-time step.

`time` (batch-time) is always the **last** axis of every tensor in `x` and of
the `forward` output, so it acts as the batch dimension: the whole time series
is processed in a single pass, and `Flux.DataLoader((x, y))` batches every
tensor along that axis consistently (nested tuples are batched element-wise).

## Interface implemented at this level

`predict`, `save_params`, and `load_params!` are provided here and are inherited
by all subtypes without modification.

## Required customisation points

| Method | Signature | Purpose |
|---|---|---|
| `preprocess`    | `(m::M, input::Dict{String,TimeSeries}) -> (Tuple, Dict{String,TimeSeries})` | Build input tensor(s) as a tuple and pre-allocate output |
| `forward`       | `(m::M, x::Tuple) -> Array{Float32,2}` | Run Flux forward pass; return `(locations, time)` |
| `postprocess!`  | `(output::Dict{String,TimeSeries}, m::M, y::AbstractMatrix)` | Fill pre-allocated output in-place from 2-D `y` |
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

`preprocess` returns both the input `x` (a tuple of tensors, each with batch-time
as its last axis) and a pre-allocated output `Dict{String, TimeSeries}` with the
correct metadata (times, station names, coordinates) and zero-initialised values.
`forward` runs the Flux model and returns a 2-D `(locations, time)` array.
`postprocess!` fills the pre-allocated output values in-place.
"""
function predict(model::AbstractFluxModel, input::Dict{String, TimeSeries})
    x, output = preprocess(model, input)
    y = forward(model, x)
    postprocess!(output, model, y)
    return output
end

# ──────────────────────────────────────────────────────────────────────────────
# train_model! — one generic loop for all AbstractFluxModel families
# ──────────────────────────────────────────────────────────────────────────────

"""
    _take_last_dim(x::Tuple, idx) -> Tuple

Slice every tensor in `x` along its last (batch-time / sample) axis at indices
`idx`, returning a new tuple of materialised arrays.  Used to split a
preprocessed input tuple into train/validation portions.
"""
_take_last_dim(x::Tuple, idx) = map(a -> copy(selectdim(a, ndims(a), idx)), x)

"""
    preprocess(model, input, target) -> (x::Tuple, y)

Training-time form of `preprocess`: build the input tuple `x` **and** the target
matrix `y` already in the model's flux-output space (2-D, batch/sample as the
last axis).  This is the per-family seam the generic `train_model!` relies on;
each concrete family implements it.  Any train-time fitting (e.g. Z-score
statistics) is done here and stored in the model settings so that inference and
the validation split reuse the same values.
"""
function preprocess(model::AbstractFluxModel, input::Dict{String, TimeSeries},
                    target::Dict{String, TimeSeries})
    error("train-form preprocess(model, input, target) not implemented for $(typeof(model))")
end

"""
    train_model!(model::AbstractFluxModel, train_settings::TrainingSettings,
                 input::Dict{String, TimeSeries}, target::Dict{String, TimeSeries};
                 val_input=nothing, val_target=nothing)
        -> (Vector{Float32}, Vector{Float32})

Single generic training loop shared by **every** model family.  Minibatch Adam
over `Flux.DataLoader((x, y))`, where `x` is the input tuple from the train-form
`preprocess(model, input, target)` and the flux model is called uniformly as
`m(x)`.  Handles per-epoch train/val RMSE, `params_best.jld2` on best val,
epoch checkpoints, and learning-rate decay.

Validation: if `val_input`/`val_target` are given they are preprocessed directly
(and take priority over `validation_split`); otherwise the last
`validation_split` fraction of the last axis is held out via `_take_last_dim`.

On first call, `"out_names"`, `"out_lons"`, `"out_lats"`, and `"out_quantity"`
are populated from the first `TimeSeries` in `target`.  Returns
`(train_losses, val_losses)` per epoch; `val_losses` is empty when there is no
validation data.
"""
function train_model!(model::AbstractFluxModel, train_settings::TrainingSettings,
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

    # Build train tensors (per-family train-form preprocess). y is 2-D with the
    # batch/sample axis last; x is the input tuple sharing that last axis.
    x_full, y_full = preprocess(model, input, target)

    # Validation: explicit data takes priority over the fraction split. The val
    # preprocess runs AFTER the train one, so any fitted stats (Z-score) are
    # already stored and reused.
    if !isnothing(val_input)
        x_val, y_val = preprocess(model, val_input, val_target)
        x, y         = x_full, y_full
        has_val      = true
    else
        nfull   = size(y_full, ndims(y_full))
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

    flux_model = get_flux_model(model)
    opt_state  = Flux.setup(Adam(train_settings.learning_rate), flux_model)
    current_lr = Float64(train_settings.learning_rate)
    # DataLoader batches the nested tuple ((x1,…,xN), y) element-wise along the
    # last axis, yielding (xb::Tuple, yb) each iteration. Flux model is called as
    # m(xb) — every flux model is callable on its input tuple.
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

# ──────────────────────────────────────────────────────────────────────────────
# Location alignment helper
# ──────────────────────────────────────────────────────────────────────────────

"""
    _check_and_align_locations(ts, expected_names, label) -> TimeSeries

Return `ts` with locations reordered to match `expected_names`.

- Extra locations in `ts` are silently dropped.
- If any name in `expected_names` is absent from `ts`, an informative error is thrown.

`label` is included in the error message to identify the source (e.g. `"input[\\"stress_x\\"]"`).
"""
function _check_and_align_locations(ts::TimeSeries,
                                    expected_names::Vector{String},
                                    label::String)
    available   = get_names(ts)
    missing_loc = setdiff(expected_names, available)
    if !isempty(missing_loc)
        error("""$label: expected locations are missing.
  expected : $expected_names
  available: $available
  missing  : $(collect(missing_loc))""")
    end
    return select_locations_by_names(ts, expected_names)
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
        -> (Tuple, Dict{String, TimeSeries})

Map `input` to model-specific input tensor(s) and a pre-allocated output container.

Returns a tuple `(x, output)` where:
- `x` is a **tuple of tensors** in the layout this model's Flux layers need
  (a 1-tuple for single-input models). Batch-time is the last axis of every
  tensor so `Flux.DataLoader` can batch them consistently.
- `output` is a `Dict{String, TimeSeries}` with the correct output metadata
  (variable names, station names, coordinates, time axis) and zero-initialised
  `values` matrices.  `postprocess!` will fill these in-place.

Responsibilities:
- Select and order input variables and locations.
- Apply input scaling or normalisation.
- Assemble the lagged input window (`time_lag = 1` means no lag).
- Arrange the data into the model's preferred tensor layout(s).
- Allocate the output `TimeSeries` objects with `zeros(Float32, ...)` values.

Must be implemented for each concrete model type.
"""
function preprocess(model::AbstractFluxModel, input::Dict{String, TimeSeries})
    error("preprocess not implemented for $(typeof(model))")
end

"""
    forward(model::AbstractFluxModel, x::Tuple) -> AbstractMatrix

Run the Flux forward pass on the input tuple `x` produced by `preprocess` and
return the model's raw 2-D output.

The flux model is called with the tuple as a **single argument**
(`get_flux_model(model)(x)`) — every flux model in the package is callable on its
input tuple (single-input Dense/Conv chains prepend `only` to unwrap the 1-tuple;
multi-arg models carry a `(m)(x::Tuple) = m(x...)` method). This one generic
definition therefore serves all families; `postprocess!` maps the raw output into
the `Dict{String, TimeSeries}` result.
"""
forward(model::AbstractFluxModel, x::Tuple) = get_flux_model(model)(x)

"""
    postprocess!(output::Dict{String, TimeSeries}, model::AbstractFluxModel,
                 y::AbstractMatrix)

Fill the pre-allocated `output` with values from the 2-D Flux output `y` of
shape `(locations, time)`.

`output` is the dict returned by `preprocess`; its `TimeSeries` values matrices
already have the right shape and can be written to with `.=`.  Apply any inverse
scaling here before writing.

Must be implemented for each concrete model type.
"""
function postprocess!(output::Dict{String, TimeSeries}, model::AbstractFluxModel,
                      y)
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
    write_outputs(model::AbstractFluxModel, data::Dict, all_settings::Dict)

Generate outputs for all entries in `all_settings["output_settings"]["outputs"]`.
`data` is the dict returned by `load_data`.  Each entry selects a split (and
optionally a `timerange` sub-window) and controls which outputs to produce.

`all_settings` is the full settings dict so that other sections (e.g.
`model_settings`, `train_settings`) are available for summary logging.

See `docs/output_settings.md` for the full schema and defaults.
"""
function write_outputs(model::AbstractFluxModel, data::Dict, all_settings::Dict)
    output_settings = get(all_settings, "output_settings", Dict{String,Any}())
    save_dir = get_settings(model)["model_dir"]
    outputs  = get(output_settings, "outputs",
                   [Dict{String,Any}("split" => "test")])

    for entry in outputs
        split = entry["split"]
        haskey(data, split) || continue

        name      = get(entry, "name",      split)
        timerange = get(entry, "timerange", nothing)

        do_timeseries     = get(entry, "timeseries",      split == "testing")
        do_fft            = get(entry, "fft",             false)
        do_scatter        = get(entry, "scatter",         false)
        do_stats          = get(entry, "write_stats",     split == "testing")
        do_series         = get(entry, "write_series",    false)
        do_tidal_analysis = get(entry, "tidal_analysis",  false) &&
                            model isa AbstractTideModel
        do_residuals      = get(entry, "residuals",       false)
        residual_path     = get(entry, "residual_path",   nothing)

        if do_timeseries || do_fft || do_scatter || do_stats || do_series ||
                do_tidal_analysis || do_residuals
            out = predict(model, data[split].input)

            if do_timeseries
                subdir = joinpath(save_dir, "$(name)_timeseries")
                mkpath(subdir)
                _plot_station_series(out, data[split].target, subdir;
                                     timerange = timerange)
            end

            if do_fft
                subdir = joinpath(save_dir, "$(name)_fft")
                mkpath(subdir)
                _plot_station_fft(out, data[split].target, subdir;
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

            if do_residuals
                out_key  = first(keys(out))
                ts_pred  = out[out_key]
                ts_true  = data[split].target[out_key]
                t_start  = get_times(ts_pred)[1]
                t_end    = get_times(ts_pred)[end]
                ts_true  = select_timespan(ts_true, t_start, t_end)
                if !isnothing(timerange)
                    ts_pred = select_timespan(ts_pred, timerange[1], timerange[2])
                    ts_true = select_timespan(ts_true, timerange[1], timerange[2])
                end
                resid_vals = get_values(ts_true) .- get_values(ts_pred)
                ts_resid   = TimeSeries(
                    resid_vals,
                    get_times(ts_true),
                    get_names(ts_true),
                    get_longitudes(ts_true),
                    get_latitudes(ts_true),
                    get_quantity(ts_pred) * "_residual",
                    get_source(ts_true),
                )
                resid_path = isnothing(residual_path) ?
                    joinpath(save_dir, "residual_$(name).jld2") :
                    residual_path
                isfile(resid_path) && rm(resid_path)
                write_to_jld2(ts_resid, resid_path)
                @info "Residual written to $resid_path"
            end

            if do_tidal_analysis
                subdir = joinpath(save_dir, "$(name)_tidal_analysis")
                mkpath(subdir)
                clist_spec = get(output_settings, "tidal_analysis_constituents", "year")
                clist = clist_spec isa Vector ? clist_spec : constituent_list(clist_spec)
                max_c = get(output_settings, "tidal_analysis_max_constituents", 20)
                _plot_station_tidal_analysis(out, data[split].target, subdir;
                                             const_list       = clist,
                                             max_constituents = max_c,
                                             timerange        = timerange)
            end
        end
    end

    if get(output_settings, "write_summary", true)
        settings = get_settings(model)
        summary  = Dict{String,Any}()
        run_info = get(all_settings, "run_info", Dict{String,Any}())
        summary["runid"]          = get(run_info, "runid", "")
        summary["description"]    = get(run_info, "description", "")
        summary["model_name"]     = settings["model_name"]
        summary["out_quantities"] = settings["out_quantities"]
        summary["n_params"]       = sum(length, Flux.trainables(get_flux_model(model)))
        if haskey(output_settings, "train_time_s")
            summary["train_time_s"] = output_settings["train_time_s"]
        end

        for entry in outputs
            split = entry["split"]
            haskey(data, split) || continue
            name      = get(entry, "name", split)
            timerange = get(entry, "timerange", nothing)

            t0  = time()
            out = predict(model, data[split].input)
            summary["predict_time_$(name)_s"] = round(time() - t0; digits=3)

            out_key  = first(keys(out))
            ts_pred  = out[out_key]
            ts_true  = data[split].target[out_key]
            t_start  = get_times(ts_pred)[1]
            t_end    = get_times(ts_pred)[end]
            ts_true  = select_timespan(ts_true, t_start, t_end)
            if !isnothing(timerange)
                ts_pred = select_timespan(ts_pred, timerange[1], timerange[2])
                ts_true = select_timespan(ts_true, timerange[1], timerange[2])
            end
            errors   = get_values(ts_true) .- get_values(ts_pred)
            summary["rmse_$(name)"] = round(sqrt(mean(abs2, errors)); digits=6)
        end

        toml_write(joinpath(save_dir, "summary.toml"), summary; overwrite=true)
    end

    return nothing
end
