"""
    train(input_toml::String; on_existing_run::Symbol=:error)

Run the full training pipeline from a TOML settings file.

Reads `[model_settings]`, `[train_settings]`, `[data_settings]`, and
optionally `[output_settings]` from `input_toml`, then:
1. Loads data via `load_data`.
2. Validates and augments settings (derives `model_dir`, location metadata).
3. Creates the model via `create_model`.
4. Trains via `train_model!`.
5. Saves weights and settings to `model_dir`.
6. Calls `write_outputs` to produce any requested plots.

Relative paths in the TOML (data files, `model_dir`) are resolved relative
to the directory containing `input_toml`, so the TOML is portable regardless
of where Julia is invoked from.

If `model_dir` already exists (from a previous run), `on_existing_run`
decides what happens, *before* anything in it is touched:
- `:error` (default) — raise an error naming the existing directory, rather
  than silently continuing or overwriting it.
- `:continue` — proceed as-is; if a weights file is present it gets loaded
  and training continues from it (see the pretrained-weights check below).
- `:overwrite` — delete any `params*` files in `model_dir` first (weights
  and epoch checkpoints), then train from scratch as if the directory were
  new.
"""
function train(input_toml::String; on_existing_run::Symbol=:error)
    on_existing_run in (:error, :continue, :overwrite) || error(
        "train: on_existing_run must be :error, :continue, or :overwrite " *
        "(got $(repr(on_existing_run)))")

    settings_file = abspath(input_toml)
    isfile(settings_file) || error("Settings file not found: $settings_file")
    toml_dir = dirname(settings_file)

    all_settings   = toml_read(settings_file)
    check_format_version(all_settings)
    model_settings = all_settings["model_settings"]
    train_settings = TrainingSettings(all_settings["train_settings"])

    # Resolve data file paths relative to the TOML location
    for f in all_settings["data_settings"]["files"]
        if haskey(f, "path") && !isabspath(f["path"])
            f["path"] = joinpath(toml_dir, f["path"])
        end
    end

    data = load_data(all_settings["data_settings"])
    haskey(data, "training") || error(
        "train: no \"training\" split found in data_settings. Found split(s): " *
        join(sort!(collect(keys(data))), ", ") *
        ". At least one file entry must have split = \"training\".")
    train_input  = data["training"].input
    train_target = data["training"].target

    validate_and_augment_settings!(all_settings, train_input, train_target)

    # Resolve model_dir relative to TOML location (may have just been derived)
    if !isabspath(model_settings["model_dir"])
        model_settings["model_dir"] = joinpath(toml_dir, model_settings["model_dir"])
    end

    save_dir = model_settings["model_dir"]

    # Decide up front, before anything below writes to save_dir, whether an
    # existing run there should block, be resumed, or be cleared. See
    # on_existing_run in the docstring.
    if isdir(save_dir)
        if on_existing_run == :error
            error("A run already exists at $save_dir. Re-run with --continue to " *
                  "resume from it, or --overwrite to discard it and train from " *
                  "scratch (on_existing_run=:continue/:overwrite if calling " *
                  "train() directly).")
        elseif on_existing_run == :overwrite
            for f in readdir(save_dir)
                startswith(f, "params") && rm(joinpath(save_dir, f))
            end
        end
    end

    mkpath(save_dir)
    toml_write(joinpath(save_dir, "run_settings.toml"), all_settings; overwrite=true)

    model = create_model(model_settings, train_input)

    # Load pre-trained weights (on_existing_run=:continue) if a weights file
    # is present -- absent after :overwrite, since that just cleared it above.
    pretrained = joinpath(save_dir, model_settings["model_weights"])
    if isfile(pretrained)
        load_params!(model, pretrained)
        @info "Continuing training from $pretrained"
    end

    t0 = time()
    val_input  = haskey(data, "validation") ? data["validation"].input  : nothing
    val_target = haskey(data, "validation") ? data["validation"].target : nothing
    train_losses, val_losses = train_model!(model, train_settings, train_input, train_target;
                                            val_input=val_input, val_target=val_target)
    train_time_s = round(time() - t0; digits=1)

    save_params(model, joinpath(save_dir, "params.jld2"); overwrite=true)

    # Point model_weights at the best-val checkpoint when one was written
    if isfile(joinpath(save_dir, "params_best.jld2"))
        model_settings["model_weights"] = "params_best.jld2"
    end

    toml_write(joinpath(save_dir, "model_settings.toml"), get_settings(model); overwrite=true)
    save_loss_plot(joinpath(save_dir, "losses.png"), train_losses, val_losses; overwrite=true)

    output_settings = get!(all_settings, "output_settings", Dict{String,Any}())
    output_settings["train_time_s"] = train_time_s
    write_outputs(model, data, all_settings)

    return nothing
end
