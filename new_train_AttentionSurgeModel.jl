# This script trains an AttentionSurgeModel on the Schureman 2011 surge dataset,
# using ERA5 wind stress and pressure as input.

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
name     = "AttentionSurgeModel"
save_dir = joinpath("models", name)

rm(save_dir, recursive=true, force=true)
mkpath(save_dir)

model_settings = Dict{String, Any}(
    "model_name" => name,
    "model_dir"  => save_dir,
    "nlags"      => 16,
    "model_pars" => Dict{String, Any}(
        "nembed"         => 32,
        "theta"          => 1000.0,
        "nheads"         => 4,
        "nlayers_branch" => 2,
        "nlayers_trunk"  => 2,
        "nhidden_trunk"  => 32,
    ),
)

train_settings = TrainingSettings(
    nepochs          = 200,       # change to e.g. 200 for a real run
    nbatches         = 64,
    learning_rate    = 1.0e-3,
    lr_decay_factor  = 0.1,
    lr_decay_rate    = 400,
    patience         = 5,
    validation_split = 0.2,
    val_daterange    = ["2012-01-01T00:00:00", "2012-01-15T00:00:00"],
)

# ──────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────
labels = ["training", "testing"]
data = Dict()
for label in labels
    @show label
    ts_h       = NetCDFTimeSeries(filenames[label]["waterlevel"], "surge")
    ts_stress_x = JLD2TimeSeries(filenames[label]["wind"], varname="stress_x")
    ts_stress_y = JLD2TimeSeries(filenames[label]["wind"], varname="stress_y")
    ts_press    = JLD2TimeSeries(filenames[label]["wind"], varname="pressure")
    t_start = max(get_times(ts_h)[1],    get_times(ts_stress_x)[1])
    t_end   = min(get_times(ts_h)[end],  get_times(ts_stress_x)[end])
    ts_h        = select_timespan(ts_h,        t_start, t_end)
    ts_stress_x = select_timespan(ts_stress_x, t_start, t_end)
    ts_stress_y = select_timespan(ts_stress_y, t_start, t_end)
    ts_press    = select_timespan(ts_press,     t_start, t_end)
    data[label] = Dict{String, TimeSeries}(
        "waterlevel" => ts_h,
        "stress_x"   => ts_stress_x,
        "stress_y"   => ts_stress_y,
        "pressure"   => ts_press,
    )
end

model_settings["nstations"] = length(get_names(data["training"]["waterlevel"]))
model_settings["nwind"]     = length(get_names(data["training"]["stress_x"]))

# ──────────────────────────────────────────────
# Build graph network (wind → waterlevel stations)
# ──────────────────────────────────────────────
wind_ts  = data["training"]["stress_x"]
surge_ts = data["training"]["waterlevel"]

in_points  = collect(zip(get_latitudes(wind_ts),  get_longitudes(wind_ts)))
out_points = collect(zip(get_latitudes(surge_ts), get_longitudes(surge_ts)))
gn = GraphNetwork(in_points, out_points)

# ──────────────────────────────────────────────
# Split into input (forcing) and target (surge)
# ──────────────────────────────────────────────
train_input  = Dict{String, TimeSeries}(k => data["training"][k] for k in ("stress_x","stress_y","pressure"))
train_target = Dict{String, TimeSeries}("surge" => data["training"]["waterlevel"])
test_input   = Dict{String, TimeSeries}(k => data["testing"][k]  for k in ("stress_x","stress_y","pressure"))
test_target  = Dict{String, TimeSeries}("surge" => data["testing"]["waterlevel"])

# ──────────────────────────────────────────────
# Create and train model
# ──────────────────────────────────────────────
model = AttentionSurgeModel(model_settings, gn)

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
