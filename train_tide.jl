# Train a tide model on the Schureman 2011 tide dataset.
#
# Select which model to train by changing model_type below:
#   "DeepONetTideModel" — branch/trunk DeepONet architecture
#   "ProductTideModel"  — multiplicative station × Doodson product with gating layers
#
# The model is astronomically driven — inputs are Doodson numbers computed from
# time and station coordinates.  Both input and target are the same waterlevel
# TimeSeries.

model_type  = "DeepONetTideModel"   # "DeepONetTideModel" | "ProductTideModel"
runid       = "dummy"
description = "Reference case trained on 2011 Schureman tides."

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
data_dir = joinpath(@__DIR__, "test_data")
data_settings = Dict{String,Any}(
    "files" => [
        Dict("path"      => joinpath(data_dir, "tides_schureman_2011.nc"),
             "format"    => "netcdf",
             "split"     => "training",
             "variables" => ["waterlevel"]),
        Dict("path"      => joinpath(data_dir, "tides_schureman_2012.nc"),
             "format"    => "netcdf",
             "split"     => "testing",
             "variables" => ["waterlevel"]),
    ],
    "model_io" => Dict("input" => ["waterlevel"], "target" => ["waterlevel"]),
)

# ──────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────
data = load_data(data_settings)
# shorthands for train/test splits
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
    "freqs"      => ["SSA","K1","O1","Q1","P1","M2","S2","N2","K2","H"],
)

if model_type == "DeepONetTideModel"
    model_settings["model_pars"] = Dict{String, Any}(
        "nlayers_branch" => 2,
        "nhidden_branch" => 64,
        "nlayers_trunk"  => 2,
        "nhidden_trunk"  => 32,
        "nlayers_down"   => 1,
    )
elseif model_type == "ProductTideModel"
    model_settings["model_pars"] = Dict{String, Any}(
        "nfeats"  => 64,
        "nlayers" => 4,
    )
end

# ──────────────────────────────────────────────
# Training settings
# ──────────────────────────────────────────────
train_settings = TrainingSettings(
    nepochs          = 500,
    nbatches         = 64,
    learning_rate    = 5.0e-3,
    lr_decay_factor  = 0.1,
    lr_decay_rate    = 400,
    patience         = 5,
    validation_split = 0.2,
)

# ──────────────────────────────────────────────
# Augmented model settings (from data) + save
# ──────────────────────────────────────────────
first_target = first(values(train_target))
get!(model_settings, "out_quantities", collect(keys(train_target)))
get!(model_settings, "out_names",      get_names(first_target))
get!(model_settings, "out_lons",       get_longitudes(first_target))
get!(model_settings, "out_lats",       get_latitudes(first_target))
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
if model_type == "DeepONetTideModel"
    model = DeepONetTideModel(model_settings)
elseif model_type == "ProductTideModel"
    model = ProductTideModel(model_settings)
else
    error("Unknown model_type: $model_type")
end

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
plot_series(model, test_input, test_target, "test")
