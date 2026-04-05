cd(@__DIR__)

using Pkg
Pkg.activate(".")

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
    "validation" => Dict(
        "waterlevel" => joinpath(data_dir, "surge_schureman_2012.nc"),
        "wind"       => joinpath(data_dir, "era5_wind_stress_2012_validation.jld2"),
    ),
)

# ──────────────────────────────────────────────
# Model / training settings
# ──────────────────────────────────────────────
model_pars = Dict(
    "theta"          => 10000.0,
    "nheads"         => 4,
    "nlayers_branch" => 2,
    "nlayers_trunk"  => 0,
    "nhidden_trunk"  => 16,
    "nembed"         => 16,
)

learning_rate   = 1.0e-3
lr_decay_factor = 0.1
lr_decay_rate   = 400
nepochs         = 2   # TESTING ONLY — change to 200 for a real run
# nepochs       = 200
checkpoints     = [40, 80, 120, 160]
val_range       = ["2012-01-01T00:00:00", "2012-01-15T00:00:00"]
nlags           = 16

name     = "TestSurgeModel"
save_dir = joinpath("models", name)

rm(save_dir, recursive=true, force=true)
mkpath(save_dir)

settings = SurgeSettings(
    model_name      = name,
    nepochs         = nepochs,
    checkpoints     = checkpoints,
    val_daterange   = val_range,
    learning_rate   = learning_rate,
    lr_decay_factor = lr_decay_factor,
    lr_decay_rate   = lr_decay_rate,
    model_dir       = save_dir,
    use_gpu         = false,
    nlags           = nlags,
    model_pars      = model_pars,
)

# ──────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────
labels = ["training", "validation", "testing"]
data = Dict()
for label in labels
    @show label
    ts_h      = NetCDFTimeSeries(filenames[label]["waterlevel"], "surge")
    ts_wind_x = JLD2TimeSeries(filenames[label]["wind"], varname="stress_x")
    ts_wind_y = JLD2TimeSeries(filenames[label]["wind"], varname="stress_y")
    ts_press  = JLD2TimeSeries(filenames[label]["wind"], varname="pressure")
    # Align to the overlapping time range (surge and ERA5 may differ by one step)
    t_start = max(get_times(ts_h)[1],   get_times(ts_wind_x)[1])
    t_end   = min(get_times(ts_h)[end], get_times(ts_wind_x)[end])
    ts_h      = select_timespan(ts_h,      t_start, t_end)
    ts_wind_x = select_timespan(ts_wind_x, t_start, t_end)
    ts_wind_y = select_timespan(ts_wind_y, t_start, t_end)
    ts_press  = select_timespan(ts_press,  t_start, t_end)
    data[label] = Dict(
        "waterlevel" => ts_h,
        "wind_x"     => ts_wind_x,
        "wind_y"     => ts_wind_y,
        "pressure"   => ts_press,
    )
end

settings.nstations = length(get_names(data["training"]["waterlevel"]))
settings.nwind     = length(get_names(data["training"]["wind_x"]))

# ──────────────────────────────────────────────
# Build graph network (connects ERA5 → surge stations)
# ──────────────────────────────────────────────
lats_in  = get_latitudes(data["training"]["wind_x"])
lons_in  = get_longitudes(data["training"]["wind_x"])
lats_out = get_latitudes(data["training"]["waterlevel"])
lons_out = get_longitudes(data["training"]["waterlevel"])

in_points  = collect(zip(lats_in,  lons_in))
out_points = collect(zip(lats_out, lons_out))

gn = GraphNetwork(in_points, out_points, max_distance=1e5)

# ──────────────────────────────────────────────
# Create and train model
# ──────────────────────────────────────────────
model = SurgeModel(gn, settings)

model, acc_losses, train_losses, test_losses =
    train_model(model, settings, data["training"], data["testing"])

save_model(model, settings)
save_settings(settings)

plot_losses(train_losses, test_losses, settings)

# ──────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────
plot_series(model, settings, data["testing"],  "testing_14d",
    timerange=settings.val_daterange)
plot_series(model, settings, data["training"], "training_14d",
    timerange=["2011-01-01T00:00:00", "2011-01-15T00:00:00"])
