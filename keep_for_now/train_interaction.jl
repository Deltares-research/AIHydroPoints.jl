# Train a tide-surge interaction model using the Schureman 2011 tide and surge
# test datasets.
#
# PURPOSE OF THIS MODEL
# The interaction model is intended to capture non-linear tide-surge interaction:
# the part of the waterlevel signal that cannot be explained by adding a linear
# tide model and a linear surge model.  In production the target should therefore
# be the surge residual:
#
#   target = observed_waterlevel - tide_prediction - linear_surge_prediction
#
# PLACEHOLDER TARGET USED HERE
# A proper residual target requires observed waterlevel data and a well-trained
# surge model, which are not yet available.  As a placeholder the surge file is
# loaded twice: once as the "surge" input and once as the "interaction" target.
# This lets the pipeline run end-to-end but the model learns a trivial mapping
# and results are not meaningful.

model_type  = "ConvInteractionModel"
runid       = "dummy"
description = "Placeholder run: surge used as interaction target (no real residual signal)."

# ──────────────────────────────────────────────
# Set up environment and load dependencies
# ──────────────────────────────────────────────
cd(@__DIR__)
using Pkg
Pkg.activate(".")
ENV["GKSwstype"] = "nul"   # to allow plotting in headless environments (e.g. remote servers, CI)
using AIHydroPoints

# ─────────────────────────────────────────────
# Create output folder
# ─────────────────────────────────────────────
save_dir = joinpath("training_output", "$(runid)_$(model_type)")
rm(save_dir, recursive=true, force=true)
mkpath(save_dir)

# ──────────────────────────────────────────────
# Data settings
# ──────────────────────────────────────────────
# The surge file is loaded twice per split: once as "surge" (input) and once as
# "interaction" (target placeholder).  Replace the interaction entries with a
# real residual file when available.
data_dir = joinpath(@__DIR__, "test_data")
data_settings = Dict{String,Any}(
    "files" => [
        # ── Training split ─────────────────────────────────────────────────
        Dict("path"      => joinpath(data_dir, "tides_schureman_2011.nc"),
             "format"    => "netcdf",
             "split"     => "training",
             "variables" => [Dict("name" => "waterlevel", "as" => "tide")]),
        Dict("path"      => joinpath(data_dir, "surge_schureman_2011.nc"),
             "format"    => "netcdf",
             "split"     => "training",
             "variables" => ["surge"]),
        Dict("path"      => joinpath(data_dir, "surge_schureman_2011.nc"),
             "format"    => "netcdf",
             "split"     => "training",
             "variables" => [Dict("name" => "surge", "as" => "interaction")]),
        # ── Testing split ──────────────────────────────────────────────────
        Dict("path"      => joinpath(data_dir, "tides_schureman_2012.nc"),
             "format"    => "netcdf",
             "split"     => "testing",
             "variables" => [Dict("name" => "waterlevel", "as" => "tide")]),
        Dict("path"      => joinpath(data_dir, "surge_schureman_2012.nc"),
             "format"    => "netcdf",
             "split"     => "testing",
             "variables" => ["surge"]),
        Dict("path"      => joinpath(data_dir, "surge_schureman_2012.nc"),
             "format"    => "netcdf",
             "split"     => "testing",
             "variables" => [Dict("name" => "surge", "as" => "interaction")]),
    ],
    "model_io" => Dict("input" => ["tide", "surge"], "target" => ["interaction"]),
)

# ──────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────
data = load_data(data_settings)
train_input  = data["training"].input
train_target = data["training"].target
test_input   = data["testing"].input
test_target  = data["testing"].target

# ──────────────────────────────────────────────
# Model settings
# ──────────────────────────────────────────────
model_settings = Dict{String, Any}(
    "model_name" => model_type,
    "model_dir"  => save_dir,
    "nlags"      => 16,
    "model_pars" => Dict{String, Any}("channels" => [64, 32, 16, 1]),
)

# ──────────────────────────────────────────────
# Training settings
# ──────────────────────────────────────────────
train_settings = TrainingSettings(
    nepochs          = 2,    # increase for a real run (e.g. 200)
    nbatches         = 64,
    learning_rate    = 1e-3,
    validation_split = 0.2,
)

# ──────────────────────────────────────────────
# Validate, augment settings (from data) + save
# ──────────────────────────────────────────────
all_settings = Dict{String,Any}(
    "run_info"       => Dict("runid" => runid, "description" => description),
    "model_settings" => model_settings,
    "train_settings" => to_dict(train_settings),
    "data_settings"  => data_settings,
)
validate_and_augment_settings!(all_settings, train_input, train_target)
toml_write(joinpath(save_dir, "run_settings.toml"), all_settings; overwrite=true)

# ──────────────────────────────────────────────
# Create model
# ──────────────────────────────────────────────
model = create_model(model_settings, train_input)

# ──────────────────────────────────────────────
# Train
# ──────────────────────────────────────────────
train_losses, val_losses = train_model!(model, train_settings, train_input, train_target)

# ──────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────
save_params(model, joinpath(save_dir, "params.jld2"); overwrite=true)
toml_write(joinpath(save_dir, "model_settings.toml"), get_settings(model); overwrite=true)

# ──────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────
save_loss_plot(joinpath(save_dir, "losses.png"), train_losses, val_losses; overwrite=true)
write_outputs(model, data, Dict{String,Any}("plot_train" => true))
