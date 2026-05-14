"""
    predict(input_toml::String)

Run inference from a TOML settings file using a previously trained model.

Reads `[model_settings]` (requires `model_dir`), `[data_settings]`, and
optionally `[output_settings]` from `input_toml`, then:
1. Loads trained model settings from `model_dir/model_settings.toml`.
2. Reconstructs the model and loads weights from `model_dir/params.jld2`.
3. Loads data via `load_data`.
4. Calls `write_outputs` to produce any requested outputs.

Outputs are written to `predict_output/<runid>_<model_name>/` relative to the
TOML file, keeping them separate from the training output directory.

Relative paths in the TOML (data files, `model_dir`) are resolved relative
to the directory containing `input_toml`, so the TOML is portable regardless
of where Julia is invoked from.
"""
function predict(input_toml::String)
    settings_file = abspath(input_toml)
    isfile(settings_file) || error("Settings file not found: $settings_file")
    toml_dir = dirname(settings_file)

    all_settings = toml_read(settings_file)

    # Resolve model_dir relative to TOML location
    model_dir = all_settings["model_settings"]["model_dir"]
    if !isabspath(model_dir)
        model_dir = joinpath(toml_dir, model_dir)
    end
    isdir(model_dir) || error("model_dir not found: $model_dir")

    # Resolve data file paths relative to TOML location
    for f in all_settings["data_settings"]["files"]
        if haskey(f, "path") && !isabspath(f["path"])
            f["path"] = joinpath(toml_dir, f["path"])
        end
    end

    model_settings = toml_read(joinpath(model_dir, "model_settings.toml"))
    model = create_model(model_settings, Dict{String,TimeSeries}())
    weights_file = joinpath(model_dir, get(model_settings, "model_weights", "params.jld2"))
    isfile(weights_file) || error("Weights file not found: $weights_file")
    load_params!(model, weights_file)

    # Derive a predict-specific output dir so outputs never land in training_output
    runid      = get(get(all_settings, "run_info", Dict()), "runid", "predict")
    model_name = model_settings["model_name"]
    predict_dir = joinpath(toml_dir, "predict_output", "$(runid)_$(model_name)")
    mkpath(predict_dir)
    model_settings["model_dir"] = predict_dir

    data = load_data(all_settings["data_settings"])

    write_outputs(model, data, all_settings)

    return nothing
end
