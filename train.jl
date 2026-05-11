# train.jl
#
# Generic training script.  Reads all settings from a TOML file and runs the
# full training pipeline: load data → validate/augment settings → create model
# → train → save → plot.
#
# Usage:
#   pixi run julia --project train.jl path/to/settings.toml
#
# The TOML must contain four top-level tables:
#   [run_info]        — runid, description
#   [model_settings]  — model_name, model_dir (optional), model-specific keys
#   [train_settings]  — nepochs, nbatches, learning_rate, ...
#   [data_settings]   — files, model_io
#
# Optional table:
#   [output_settings] — plot_train (default false), plot_test (default true)

length(ARGS) == 1 ||
    error("Usage: julia train.jl <settings.toml>\nGot ARGS = $(ARGS)")
settings_file = abspath(ARGS[1])   # resolve before cd changes the working dir
isfile(settings_file) ||
    error("Settings file not found: $settings_file")

cd(@__DIR__)
using Pkg
Pkg.activate(".")
ENV["GKSwstype"] = "nul"   # allow plotting in headless environments
using AIHydroPoints

# ── Load settings ─────────────────────────────────────────────────────────────
all_settings   = toml_read(settings_file)
model_settings = all_settings["model_settings"]
train_settings = TrainingSettings(all_settings["train_settings"])

# ── Load data ─────────────────────────────────────────────────────────────────
data         = load_data(all_settings["data_settings"])
train_input  = data["training"].input
train_target = data["training"].target

# ── Validate + augment settings (derives model_dir if absent) ─────────────────
validate_and_augment_settings!(all_settings, train_input, train_target)

# ── Create output folder and persist full settings ────────────────────────────
save_dir = model_settings["model_dir"]
mkpath(save_dir)
toml_write(joinpath(save_dir, "run_settings.toml"), all_settings; overwrite=true)

# ── Create model ──────────────────────────────────────────────────────────────
model = create_model(model_settings, train_input)

# ── Train ─────────────────────────────────────────────────────────────────────
train_losses, val_losses = train_model!(model, train_settings, train_input, train_target)

# ── Plots and other outputs────────────────────────────────────────────────────
save_params(model, joinpath(save_dir, "params.jld2"); overwrite=true)
toml_write(joinpath(save_dir, "model_settings.toml"), get_settings(model); overwrite=true)

save_loss_plot(joinpath(save_dir, "losses.png"), train_losses, val_losses; overwrite=true)

write_outputs(model, data, get(all_settings, "output_settings", Dict{String,Any}()))
