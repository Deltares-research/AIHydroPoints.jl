# input_processing.jl
#
# Generic settings validation and augmentation for training scripts.
# Called after loading data and before constructing a model.

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

    # ── Augment from target (all models) ─────────────────────────────────────
    first_target = first(values(train_target))
    get!(model_settings, "out_quantities",    collect(keys(train_target)))
    get!(model_settings, "out_names",         get_names(first_target))
    get!(model_settings, "out_lons",          get_longitudes(first_target))
    get!(model_settings, "out_lats",          get_latitudes(first_target))
    get!(model_settings, "nlocations_output", length(model_settings["out_names"]))

    # ── Augment from input (skipped for tide models with no loaded inputs) ────
    if !isempty(train_input)
        first_input = first(values(train_input))
        get!(model_settings, "in_quantities",    collect(keys(train_input)))
        get!(model_settings, "in_names",         get_names(first_input))
        get!(model_settings, "in_lons",          get_longitudes(first_input))
        get!(model_settings, "in_lats",          get_latitudes(first_input))
        get!(model_settings, "nlocations_input", length(model_settings["in_names"]))
    end

    # ── Populate output_settings defaults ────────────────────────────────────
    out = get!(all_settings, "output_settings", Dict{String,Any}())
    get!(out, "plot_train", false)
    get!(out, "plot_test",  true)
    get!(out, "plot_fft",   false)

    # ── Model-specific validation hook ───────────────────────────────────────
    validate_model_settings!(get_model_type(model_settings), model_settings)

    return nothing
end
