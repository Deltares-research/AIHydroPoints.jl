# Train a wave model on the waves_2021 test dataset (NOOS format).
#
# Select which model to train by changing model_type below:
#   "ConvWaveModel"     — WaveInputLayer + strided Conv (exp channel modulation)
#   "DeepONetWaveModel" — strided Conv branch + dot-product station merge

model_type  = "ConvWaveModel"      # "ConvWaveModel" | "DeepONetWaveModel"
model_type  = "DeepONetWaveModel"
runid       = "dummy"
description = "Reference case trained on 2021 KNMI Harmonie wind (Jan–Sep)."

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
wave_dir = joinpath(@__DIR__, "test_data", "waves_2021")
data_settings = Dict{String,Any}(
    "files" => [
        Dict("path"      => wave_dir,
             "format"    => "noos",
             "source"    => "knmi_harmonie40_wind",
             "split"     => "training",
             "timerange" => ["2021-01-01", "2021-09-30T23:00:00"],
             "variables" => ["wind_speed", "wind_direction"]),
        Dict("path"      => wave_dir,
             "format"    => "noos",
             "source"    => "swan_dcsm_harmonie",
             "split"     => "training",
             "timerange" => ["2021-01-01", "2021-09-30T23:00:00"],
             "variables" => ["wave_height"]),
        Dict("path"      => wave_dir,
             "format"    => "noos",
             "source"    => "knmi_harmonie40_wind",
             "split"     => "testing",
             "timerange" => ["2021-10-01", "2021-12-31T23:00:00"],
             "variables" => ["wind_speed", "wind_direction"]),
        Dict("path"      => wave_dir,
             "format"    => "noos",
             "source"    => "swan_dcsm_harmonie",
             "split"     => "testing",
             "timerange" => ["2021-10-01", "2021-12-31T23:00:00"],
             "variables" => ["wave_height"]),
    ],
    "model_io" => Dict("input" => ["wind_speed", "wind_direction"], "target" => ["wave_height"]),
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
    "wind_scale" => 0.5,
    "wave_scale" => 3.0,
)

if model_type == "ConvWaveModel"
    model_settings["n_input_channels"] = 64
    model_settings["model_pars"] = Dict{String, Any}(
        "nchannel"   => [64, 64, 64, 1],
        "activation" => "swish",
    )
elseif model_type == "DeepONetWaveModel"
    model_settings["model_pars"] = Dict{String, Any}(
        "nchannel"   => [32, 32, 32, 16],
        "activation" => "swish",
    )
end

# ──────────────────────────────────────────────
# Training settings
# ──────────────────────────────────────────────
train_settings = TrainingSettings(
    nepochs          = 50,
    nbatches         = 256,
    learning_rate    = 1.0e-3,
    lr_decay_factor  = 0.1,
    lr_decay_rate    = 400,
    patience         = 5,
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
if model_type == "ConvWaveModel"
    model = ConvWaveModel(model_settings)
elseif model_type == "DeepONetWaveModel"
    model = DeepONetWaveModel(model_settings)
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
