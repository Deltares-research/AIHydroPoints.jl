# This script trains a LinearSurgeModel on the Schureman 2011 surge dataset, using ERA5 wind stress and pressure as input.
# It's a stand alone test of the new AbstractModel interface, and is not integrated into the main training.jl script yet.  The code is a bit rough and ready, but it serves to check that the new interface works as intended before we refactor the main training script.


cd(@__DIR__)

using Pkg
Pkg.activate(".")

ENV["GKSwstype"] = "nul"   # headless GR backend (no display needed)
using AIHydroPoints

# ──────────────────────────────────────────────
# File paths
# ──────────────────────────────────────────────
data_dir = joinpath(@__DIR__, "test_data")

filenames = Dict(
    "training" => Dict(
        "waterlevel" => joinpath(data_dir, "surge_schureman_2011.nc"),
        "wind"       => joinpath(data_dir, "era5_wind_stress_2011_testing.jld2"),
    ),
    "testing" => Dict(
        "waterlevel" => joinpath(data_dir, "surge_schureman_2012.nc"),
        "wind"       => joinpath(data_dir, "era5_wind_stress_2012_validation.jld2"),
    ),
)

# ──────────────────────────────────────────────
# Model / training settings
# ──────────────────────────────────────────────
name     = "LinearSurgeModel"
save_dir = joinpath("models", name)

rm(save_dir, recursive=true, force=true)
mkpath(save_dir)

model_settings = Dict{String, Any}(
    "model_name" => name,
    "model_dir"  => save_dir,
    "nlags"      => 16,
)

train_settings = TrainingSettings(
    nepochs         = 20,        # change to e.g. 200 for a real run
    nbatches        = 64,
    learning_rate   = 1.0e-3,
    lr_decay_factor = 0.1,
    lr_decay_rate   = 400,
    patience        = 5,
    validation_split = 0.2, # fraction of training data to use for validation (early stopping)
    val_daterange   = ["2012-01-01T00:00:00", "2012-01-15T00:00:00"],
)

# ──────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────
labels = ["training", "testing"]
data = Dict()
for label in labels
    @show label
    ts_h      = NetCDFTimeSeries(filenames[label]["waterlevel"], "surge")
    ts_wind_x = JLD2TimeSeries(filenames[label]["wind"], varname="stress_x")
    ts_wind_y = JLD2TimeSeries(filenames[label]["wind"], varname="stress_y")
    ts_press  = JLD2TimeSeries(filenames[label]["wind"], varname="pressure")
    t_start = max(get_times(ts_h)[1],   get_times(ts_wind_x)[1])
    t_end   = min(get_times(ts_h)[end], get_times(ts_wind_x)[end])
    ts_h      = select_timespan(ts_h,      t_start, t_end)
    ts_wind_x = select_timespan(ts_wind_x, t_start, t_end)
    ts_wind_y = select_timespan(ts_wind_y, t_start, t_end)
    ts_press  = select_timespan(ts_press,  t_start, t_end)
    data[label] = Dict{String, TimeSeries}(
        "waterlevel" => ts_h,
        "wind_x"     => ts_wind_x,
        "wind_y"     => ts_wind_y,
        "pressure"   => ts_press,
    )
end

model_settings["nstations"] = length(get_names(data["training"]["waterlevel"]))
model_settings["nwind"]     = length(get_names(data["training"]["wind_x"]))

# ──────────────────────────────────────────────
# Split into input (forcing) and target (surge)
# ──────────────────────────────────────────────
train_input  = Dict{String, TimeSeries}(k => data["training"][k]  for k in ("wind_x","wind_y","pressure"))
train_target = Dict{String, TimeSeries}("surge" => data["training"]["waterlevel"])
test_input   = Dict{String, TimeSeries}(k => data["testing"][k]   for k in ("wind_x","wind_y","pressure"))
test_target  = Dict{String, TimeSeries}("surge" => data["testing"]["waterlevel"])

# ──────────────────────────────────────────────
# Create and train model
# ──────────────────────────────────────────────
model = LinearSurgeModel(model_settings)

# train_model! populates out_names/out_lons/out_lats/out_quantity from train_target
train_losses, val_losses = train_model!(model, train_settings, train_input, train_target)

# ──────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────
save_params(model, joinpath(save_dir, "params.jld2"); overwrite=true)

toml_write(joinpath(save_dir, "settings.toml"), get_settings(model); overwrite=true)

# ──────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────
save_loss_plot(joinpath(save_dir, "losses.png"), train_losses, val_losses; overwrite=true)

# ──────────────────────────────────────────────
# Run inference on test set
# ──────────────────────────────────────────────
test_output = predict(model, test_input)
