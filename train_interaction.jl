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
# SYNTHETIC TARGET USED HERE
# The test data do not include observed waterlevel, so we synthesise a dummy
# target as tide + surge.  This lets the code run end-to-end but produces
# meaningless results: the model is asked to learn the trivial identity
# waterlevel = tide + surge, which contains no residual interaction at all.
# Do not interpret the loss values or evaluation plots as meaningful.

model_type  = "ConvInteractionModel"
runid       = "dummy"
description = "Synthetic test case: target is tide + surge (no real interaction signal)."

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
# Target waterlevel is synthesised from tide + surge below, so model_io only
# routes the raw loaded variables.  Both splits load the same two quantities.
data_dir = joinpath(@__DIR__, "test_data")
data_settings = Dict{String,Any}(
    "files" => [
        Dict("path"      => joinpath(data_dir, "tides_schureman_2011.nc"),
             "format"    => "netcdf",
             "split"     => "training",
             "variables" => [Dict("name" => "waterlevel", "as" => "tide")]),
        Dict("path"      => joinpath(data_dir, "surge_schureman_2011.nc"),
             "format"    => "netcdf",
             "split"     => "training",
             "variables" => ["surge"]),
        Dict("path"      => joinpath(data_dir, "tides_schureman_2012.nc"),
             "format"    => "netcdf",
             "split"     => "testing",
             "variables" => [Dict("name" => "waterlevel", "as" => "tide")]),
        Dict("path"      => joinpath(data_dir, "surge_schureman_2012.nc"),
             "format"    => "netcdf",
             "split"     => "testing",
             "variables" => ["surge"]),
    ],
    "model_io" => Dict("input" => ["tide", "surge"], "target" => ["tide", "surge"]),
)

# ──────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────
data = load_data(data_settings)
train_input = data["training"].input
test_input  = data["testing"].input

# Synthesise a dummy target (tide + surge) for code testing only.
# Replace with: observed_waterlevel - tide_prediction - linear_surge_prediction
function _synthesize_waterlevel(input::Dict{String, TimeSeries})
    ts_tide  = input["tide"]
    ts_surge = input["surge"]
    TimeSeries(
        Float32.(get_values(ts_tide) .+ get_values(ts_surge)),
        get_times(ts_tide),
        get_names(ts_tide),
        Float64.(get_longitudes(ts_tide)),
        Float64.(get_latitudes(ts_tide)),
        "waterlevel",
        "synthetic",
    )
end
train_target = Dict{String, TimeSeries}("waterlevel" => _synthesize_waterlevel(train_input))
test_target  = Dict{String, TimeSeries}("waterlevel" => _synthesize_waterlevel(test_input))

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
# Augmented model settings (from data) + save
# ──────────────────────────────────────────────
first_target = first(values(train_target))
first_input  = first(values(train_input))
get!(model_settings, "out_quantities", collect(keys(train_target)))
get!(model_settings, "out_names",      get_names(first_target))
get!(model_settings, "out_lons",       get_longitudes(first_target))
get!(model_settings, "out_lats",       get_latitudes(first_target))
get!(model_settings, "in_quantities",  collect(keys(train_input)))
get!(model_settings, "in_names",       get_names(first_input))
get!(model_settings, "in_lons",        get_longitudes(first_input))
get!(model_settings, "in_lats",        get_latitudes(first_input))
get!(model_settings, "nlocations_output", length(model_settings["out_names"]))

all_settings = Dict{String,Any}(
    "run_info"       => Dict("runid" => runid, "description" => description),
    "model_settings" => model_settings,
    "train_settings" => to_dict(train_settings),
    "data_settings"  => data_settings,
)
toml_write(joinpath(save_dir, "run_settings.toml"), all_settings; overwrite=true)

# ──────────────────────────────────────────────
# Create model
# ──────────────────────────────────────────────
model = ConvInteractionModel(model_settings)

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

# ──────────────────────────────────────────────
# Run inference on test set
# ──────────────────────────────────────────────
test_output = predict(model, test_input)

# ──────────────────────────────────────────────
# Evaluation plots
# ──────────────────────────────────────────────
plot_series(model, train_input, train_target, "train")
plot_series(model, test_input,  test_target,  "test")
