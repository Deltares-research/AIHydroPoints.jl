# Train a surge model on the Schureman 2011 surge dataset using ERA5 wind stress and pressure.
#
# Select which model to train by changing model_type below:
#   "LinearSurgeModel"    — single Dense layer (fast baseline)
#   "AttentionSurgeModel" — transformer branch + dense trunk + graph adjacency

model_type  = "LinearSurgeModel"   # "LinearSurgeModel" | "ConvSurgeModel" | "AttentionSurgeModel"
model_type  = "ConvSurgeModel"     # "LinearSurgeModel" | "ConvSurgeModel" | "AttentionSurgeModel"
runid       = "dummy"
description = "Reference case trained on 2011 ERA5 wind stress and Schureman surge."

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
        Dict("path"      => joinpath(data_dir, "surge_schureman_2011.nc"),
             "format"    => "netcdf",
             "split"     => "training",
             "variables" => ["surge"]),
        Dict("path"      => joinpath(data_dir, "era5_wind_stress_2011_testing.jld2"),
             "format"    => "jld2",
             "split"     => "training",
             "variables" => ["stress_x", "stress_y", "pressure"]),
        Dict("path"      => joinpath(data_dir, "surge_schureman_2012.nc"),
             "format"    => "netcdf",
             "split"     => "testing",
             "variables" => ["surge"]),
        Dict("path"      => joinpath(data_dir, "era5_wind_stress_2012_validation.jld2"),
             "format"    => "jld2",
             "split"     => "testing",
             "variables" => ["stress_x", "stress_y", "pressure"]),
    ],
    "model_io" => Dict("input" => ["stress_x", "stress_y", "pressure"], "target" => ["surge"]),
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
    "nlags"      => 16,
)

if model_type == "ConvSurgeModel"
    model_settings["model_pars"] = Dict{String, Any}(
        "channels"   => [32, 16],
        "filtersize" => 3,
    )
elseif model_type == "AttentionSurgeModel"
    model_settings["model_pars"] = Dict{String, Any}(
        "nembed"         => 32,
        "theta"          => 1000.0,
        "nheads"         => 4,
        "nlayers_branch" => 2,
        "nlayers_trunk"  => 2,
        "nhidden_trunk"  => 32,
    )
end

# ──────────────────────────────────────────────
# Training settings
# ──────────────────────────────────────────────
train_settings = TrainingSettings(
    nepochs          = 100,
    nbatches         = 64,
    learning_rate    = 1.0e-3,
    lr_decay_factor  = 0.1,
    lr_decay_rate    = 400,
    patience         = 5,
    validation_split = 0.2,
    val_daterange    = ["2012-01-01T00:00:00", "2012-01-15T00:00:00"],
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
get!(model_settings, "nstations",      length(model_settings["out_names"])) # TODO: rename to nlocations_output see plan.md
get!(model_settings, "nwind",          length(model_settings["in_names"]))  # TODO: rename to nlocations_input see plan.md

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
if model_type == "LinearSurgeModel"
    model = LinearSurgeModel(model_settings)
elseif model_type == "ConvSurgeModel"
    model = ConvSurgeModel(model_settings)
elseif model_type == "AttentionSurgeModel"
    wind_ts    = train_input["stress_x"]
    surge_ts   = train_target["surge"]
    in_points  = collect(zip(get_latitudes(wind_ts),  get_longitudes(wind_ts)))
    out_points = collect(zip(get_latitudes(surge_ts), get_longitudes(surge_ts)))
    gn    = GraphNetwork(in_points, out_points)
    model = AttentionSurgeModel(model_settings, gn)
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
