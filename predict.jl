# predict.jl
#
# Inference script.  Loads a trained model and runs predictions on new data.
# All output is controlled by the [output_settings] table.
#
# Usage:
#   pixi run julia --project predict.jl path/to/settings.toml
#
# The TOML must contain:
#   [model_settings]  — model_dir: path to the trained model directory
#   [data_settings]   — files, model_io  (same format as train.jl)
#
# Optional tables:
#   [run_info]        — runid, description (informational only)
#   [output_settings] — plot_train (default false), plot_test (default true),
#                        plot_fft (default false)
#
# The trained model settings are loaded from model_dir/model_settings.toml
# and the weights from model_dir/params.jld2.

length(ARGS) == 1 ||
    error("Usage: julia predict.jl <settings.toml>\nGot ARGS = $(ARGS)")
settings_file = abspath(ARGS[1])   # resolve before cd changes the working dir
isfile(settings_file) ||
    error("Settings file not found: $settings_file")

cd(@__DIR__)
using Pkg
Pkg.activate(".")
ENV["GKSwstype"] = "nul"   # allow plotting in headless environments
using AIHydroPoints

# ── Load inference settings ───────────────────────────────────────────────────
all_settings = toml_read(settings_file)
model_dir    = all_settings["model_settings"]["model_dir"]
isdir(model_dir) ||
    error("model_dir not found: $model_dir")

# ── Reconstruct model from training output ────────────────────────────────────
model_settings = toml_read(joinpath(model_dir, "model_settings.toml"))
model = create_model(model_settings, Dict{String,TimeSeries}())
load_params!(model, joinpath(model_dir, "params.jld2"))

# ── Load inference data ───────────────────────────────────────────────────────
data = load_data(all_settings["data_settings"])

# ── Outputs ───────────────────────────────────────────────────────────────────
# This will also trigger running the model.
write_outputs(model, data, get(all_settings, "output_settings", Dict{String,Any}()))
