"""
    train(input_toml::String)

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
"""
function train(input_toml::String)
    settings_file = abspath(input_toml)
    isfile(settings_file) || error("Settings file not found: $settings_file")
    toml_dir = dirname(settings_file)

    all_settings   = toml_read(settings_file)
    model_settings = all_settings["model_settings"]
    train_settings = TrainingSettings(all_settings["train_settings"])

    # Resolve data file paths relative to the TOML location
    for f in all_settings["data_settings"]["files"]
        if haskey(f, "path") && !isabspath(f["path"])
            f["path"] = joinpath(toml_dir, f["path"])
        end
    end

    data         = load_data(all_settings["data_settings"])
    train_input  = data["training"].input
    train_target = data["training"].target

    validate_and_augment_settings!(all_settings, train_input, train_target)

    # Resolve model_dir relative to TOML location (may have just been derived)
    if !isabspath(model_settings["model_dir"])
        model_settings["model_dir"] = joinpath(toml_dir, model_settings["model_dir"])
    end

    save_dir = model_settings["model_dir"]
    mkpath(save_dir)
    toml_write(joinpath(save_dir, "run_settings.toml"), all_settings; overwrite=true)

    model = create_model(model_settings, train_input)
    t0 = time()
    val_input  = haskey(data, "validation") ? data["validation"].input  : nothing
    val_target = haskey(data, "validation") ? data["validation"].target : nothing
    train_losses, val_losses = train_model!(model, train_settings, train_input, train_target;
                                            val_input=val_input, val_target=val_target)
    train_time_s = round(time() - t0; digits=1)

    save_params(model, joinpath(save_dir, "params.jld2"); overwrite=true)
    toml_write(joinpath(save_dir, "model_settings.toml"), get_settings(model); overwrite=true)
    save_loss_plot(joinpath(save_dir, "losses.png"), train_losses, val_losses; overwrite=true)

    output_settings = get!(all_settings, "output_settings", Dict{String,Any}())
    output_settings["train_time_s"] = train_time_s
    write_outputs(model, data, all_settings)

    return nothing
end
