cd(@__DIR__)

using Pkg
Pkg.activate(".")

using AIHydroPoints

labels=["training","validation","testing"]
tide_model = "TestTideModel"
filenames=Dict()

filenames["training"] = Dict(
    "waterlevel" => joinpath("models", tide_model, "training_surge.jld2"),
    "wind" => "era5_wind_stress_2008_training.jld2"
)
filenames["testing"] = Dict(
    "waterlevel" => joinpath("models", tide_model, "testing_surge.jld2"),
    "wind" => "era5_wind_stress_2011_testing.jld2"
)
filenames["validation"] = Dict(
    "waterlevel" => joinpath("models", tide_model, "testing_surge.jld2"),
    "wind" => "era5_wind_stress_2011_testing.jld2"
)

# Use default options for everything
# Or overrise using kwargs
model_pars = Dict()
model_pars["theta"] = 10000.0
model_pars["nheads"] = 4
model_pars["nlayers_branch"] = 7
model_pars["nlayers_trunk"] = 0
model_pars["nhidden_trunk"] = 16
model_pars["nembed"] = 32

learning_rate = 1.0e-3
lr_decay_factor = 0.1
lr_decay_rate = 75
nepochs = 200
checkpoints = [40, 80, 120, 160]
val_range = ["2011-12-01T00:00:00", "2011-12-31T23:00:00"]
nlags = 16

name = "SurgeGraphDON"
save_dir = "models/$(name)"

# rm(save_dir, recursive=true)
if !isdir(save_dir)
    mkpath(save_dir)
end

settings = SurgeSettings(
    model_name=name,
    nepochs=nepochs,
    checkpoints=checkpoints,
    val_daterange=val_range,
    learning_rate=learning_rate,
    lr_decay_factor=lr_decay_factor,
    lr_decay_rate=lr_decay_rate,
    model_dir=save_dir,
    use_gpu=true,
    nlags=nlags,
    model_pars=model_pars
)

data = Dict()
for label in labels
    @show label
    ts_h = JLD2TimeSeries(filenames[label]["waterlevel"], varname="waterlevel")
    ts_wind_x = JLD2TimeSeries(filenames[label]["wind"], varname = "stress_x")
    ts_wind_y = JLD2TimeSeries(filenames[label]["wind"], varname = "stress_y")
    ts_press = JLD2TimeSeries(filenames[label]["wind"], varname = "pressure")
    data[label] = Dict(
        "waterlevel" => ts_h,
        "wind_x" => ts_wind_x,
        "wind_y" => ts_wind_y,
        "pressure" => ts_press
    )
end

settings.nstations = length(get_names(data["training"]["waterlevel"]))
settings.nwind = length(get_names(data["training"]["wind_x"]))

lats_in = get_latitudes(data["training"]["wind_x"])
lons_in = get_longitudes(data["training"]["wind_x"])

in_points = collect(zip(lats_in, lons_in))

lats_out = get_latitudes(data["training"]["waterlevel"])
lons_out = get_longitudes(data["training"]["waterlevel"])

out_points = collect(zip(lats_out,lons_out))

gn = GraphNetwork(in_points, out_points, max_distance=1e5)

model = SurgeModel(gn, settings)

model, acc_losses, train_losses, test_losses = train_model(model, settings, data["training"], data["testing"])

save_model(model, settings)
save_settings(settings)

plot_losses(train_losses, test_losses, settings)

# Make predictions for entire data set

plot_series(model, settings, 
    data["testing"],
    "testing_14d", timerange=settings.val_daterange)
plot_series(model, settings, 
    data["training"],
    "training_14d", timerange=["2009-01-01T00:00:00", "2009-01-15T00:00:00"])