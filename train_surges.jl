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
model_pars["channels"] = [32,32,32,1]
model_pars["filter"] = 2
model_pars["stride"] = 2

learning_rate = 1.0e-3
nepochs = 100
nlags=16

name = "TestSurgeModel"
save_dir = "models/$(name)"

rm(save_dir, recursive=true)
if !isdir(save_dir)
    mkpath(save_dir)
end

settings = SurgeSettings(
    model_name=name,
    nepochs=nepochs,
    learning_rate=learning_rate,
    model_dir=save_dir,
    use_gpu=true,
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

model = create_surge_model(settings)


train_data = prepare_train_data(data["training"]["waterlevel"], data["training"]["wind_x"],
                data["training"]["wind_y"], data["training"]["pressure"], settings)
test_data = prepare_train_data(data["testing"]["waterlevel"], data["testing"]["wind_x"],
                data["testing"]["wind_y"], data["testing"]["pressure"], settings)                

model, acc_losses, train_losses, test_losses = train_model(model, settings, train_data, test_data)

save_model(model, settings)
save_settings(settings)

plot_losses(train_losses, test_losses, settings)

# Make predictions for entire data set

val_station = "VLISSGN"
start_time = "2011-01-01T00:00:00"
end_time = "2011-01-15T00:00:00"

plot_series(model, settings, 
    data["training"]["waterlevel"], data["training"]["wind_x"], data["training"]["wind_y"], data["training"]["pressure"], 
    "training", write_series=true, write_format="something")
plot_series(model, settings, 
    data["testing"]["waterlevel"], data["testing"]["wind_x"], data["testing"]["wind_y"], data["testing"]["pressure"], 
    "testing", write_series=true)
plot_series(model, settings, 
    data["testing"]["waterlevel"], data["testing"]["wind_x"], data["testing"]["wind_y"], data["testing"]["pressure"], 
    "testing_14d", timerange=[start_time, end_time])
plot_series(model, settings, 
    data["training"]["waterlevel"], data["training"]["wind_x"], data["training"]["wind_y"], data["training"]["pressure"], 
    "training_14d", timerange=["2009-01-01T00:00:00", "2009-01-15T00:00:00"])

# Or get the predicted time series itself for a single station

ts_single_h = select_location_by_name(data["validation"]["waterlevel"], val_station)
ts_single_h = select_timespan(ts_single_h, start_time, end_time)

ts_single_wind_x = select_timespan(data["validation"]["wind_x"], start_time, end_time)
ts_single_wind_y = select_timespan(data["validation"]["wind_y"], start_time, end_time)
ts_single_press = select_timespan(data["validation"]["pressure"], start_time, end_time)

y_hat = predict(model, settings, ts_single_h, ts_single_wind_x, ts_single_wind_y, ts_single_press)