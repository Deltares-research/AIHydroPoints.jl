# input_processing.jl
#
# Generic settings validation and augmentation for training scripts.
# Called after loading data and before constructing a model.

"""
    CURRENT_FORMAT_VERSION

The TOML input `format_version` this build understands. Bump when making a
breaking change to the input format and record it in
`docs/settings.md#format-versions`.
"""
const CURRENT_FORMAT_VERSION = 2

"""
    check_format_version(all_settings)

Verify the top-level `format_version`. A file with no key is treated as version 1
(the pre-`format_version` format) and rejected with a migration message; a newer
version than this build understands is also rejected. Called from `train`/`predict`
right after reading the TOML.
"""
function check_format_version(all_settings::AbstractDict)
    v = get(all_settings, "format_version", 1)
    if v < CURRENT_FORMAT_VERSION
        error("""
        This input uses TOML format version $v; the current format is version $CURRENT_FORMAT_VERSION.
        Renamed keys: nbatches→batch_size, lr_decay_rate→lr_decay_epochs, \
        patience→early_stopping_epochs, weight_reg→weight_decay; val_daterange removed; \
        output flags → plot_*/write_* (e.g. timeseries→plot_timeseries, residuals→write_residuals).
        Add `format_version = $CURRENT_FORMAT_VERSION` at the top of the file and migrate the keys.
        See docs/settings.md#format-versions.""")
    elseif v > CURRENT_FORMAT_VERSION
        error("This input declares format_version $v, but this build supports only up to " *
              "$CURRENT_FORMAT_VERSION. Update AIHydroPoints.")
    end
    return nothing
end

"""
    validate_and_augment_settings!(
        all_settings::Dict{String,Any},
        train_input::Dict{String,TimeSeries},
        train_target::Dict{String,TimeSeries},
    )

Validate the top-level settings dict and augment `all_settings["model_settings"]`
with location metadata derived from the loaded training data.

## Checks (errors on failure)

- `all_settings` has `"model_settings"` with `"model_name"` present.
- `all_settings` has `"data_settings"` with `"model_io"` containing `"input"` and `"target"`.
- `all_settings` has `"run_info"` with `"runid"`.
- `train_target` is non-empty.

## Defaults

- `model_settings["model_dir"]` is derived from `training_output/<runid>_<model_name>`
  if not already present.

## Augmentation

Populates missing keys in `model_settings` from the loaded data:

- Output-side (all models): `out_quantities`, `out_names`, `out_lons`, `out_lats`,
  `nlocations_output`.
- Input-side (models with loaded inputs): `in_quantities`, `in_names`, `in_lons`,
  `in_lats`, `nlocations_input`. Skipped when `train_input` is empty (e.g. tide models
  whose inputs are computed from time and coordinates, not loaded from files).

Model-specific required keys (e.g. `"nlags"`, `"freqs"`, `"model_pars"`) are validated
by each model constructor — not here.
"""
function validate_and_augment_settings!(
    all_settings::Dict{String,Any},
    train_input::Dict{String,TimeSeries},
    train_target::Dict{String,TimeSeries},
)
    # ── Structural checks ─────────────────────────────────────────────────────
    haskey(all_settings, "model_settings") ||
        error("validate_and_augment_settings!: missing top-level key \"model_settings\"")
    haskey(all_settings, "data_settings") ||
        error("validate_and_augment_settings!: missing top-level key \"data_settings\"")
    haskey(all_settings, "run_info") ||
        error("validate_and_augment_settings!: missing top-level key \"run_info\"")

    model_settings = all_settings["model_settings"]
    data_settings  = all_settings["data_settings"]
    run_info       = all_settings["run_info"]

    haskey(model_settings, "model_name") ||
        error("validate_and_augment_settings!: model_settings missing \"model_name\"")
    haskey(run_info, "runid") ||
        error("validate_and_augment_settings!: run_info missing \"runid\"")

    model_io = get(data_settings, "model_io", nothing)
    isnothing(model_io) &&
        error("validate_and_augment_settings!: data_settings missing \"model_io\"")
    haskey(model_io, "input") ||
        error("validate_and_augment_settings!: model_io missing \"input\"")
    haskey(model_io, "target") ||
        error("validate_and_augment_settings!: model_io missing \"target\"")

    isempty(train_target) &&
        error("validate_and_augment_settings!: train_target is empty")

    # ── Derive model_dir if absent ────────────────────────────────────────────
    if !haskey(model_settings, "model_dir")
        runid      = run_info["runid"]
        model_name = model_settings["model_name"]
        model_settings["model_dir"] = joinpath("training_output", "$(runid)_$(model_name)")
    end

    get!(model_settings, "model_weights", "params.jld2")

    # ── Augment from target (all models) ─────────────────────────────────────
    first_target = first(values(train_target))
    get!(model_settings, "out_quantities",    collect(keys(train_target)))
    get!(model_settings, "out_names",         get_names(first_target))
    get!(model_settings, "out_lons",          get_longitudes(first_target))
    get!(model_settings, "out_lats",          get_latitudes(first_target))
    get!(model_settings, "nlocations_output", length(model_settings["out_names"]))

    # ── Augment from input (skipped for tide models with no loaded inputs) ────
    if !isempty(train_input)
        # The input grid (in_names, nlocations_input) is the *forcing* grid. `tide`
        # is an input on the OUTPUT stations (surge-interaction models), so it must
        # not define the input grid — prefer any non-"tide" input as the reference.
        forcing_keys = [k for k in keys(train_input) if k != "tide"]
        ref_key      = isempty(forcing_keys) ? first(keys(train_input)) : first(forcing_keys)
        ref_input    = train_input[ref_key]
        get!(model_settings, "in_quantities",    collect(keys(train_input)))
        get!(model_settings, "in_names",         get_names(ref_input))
        get!(model_settings, "in_lons",          get_longitudes(ref_input))
        get!(model_settings, "in_lats",          get_latitudes(ref_input))
        get!(model_settings, "nlocations_input", length(model_settings["in_names"]))
    end

    # ── Populate output_settings defaults ────────────────────────────────────
    out = get!(all_settings, "output_settings", Dict{String,Any}())
    get!(out, "series_format", "netcdf")
    get!(out, "write_summary", true)
    if !haskey(out, "outputs")
        out["outputs"] = [Dict{String,Any}(
            "split"               => "testing",
            "plot_timeseries"     => true,
            "plot_fft"            => false,
            "plot_scatter"        => false,
            "scatter_add_fit"     => true,
            "scatter_add_qq"      => true,
            "write_stats"         => true,
            "write_series"        => false,
            "plot_tidal_analysis" => false,
        )]
    end

    # ── Model-specific validation hook ───────────────────────────────────────
    validate_model_settings!(get_model_type(model_settings), model_settings)

    return nothing
end
