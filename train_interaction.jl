cd(@__DIR__)

using Pkg
Pkg.activate(".")

using AIHydroPoints

labels = ["training", "validation", "testing"]
tide_model = "TestTideModel"
surge_model = "TestSurgeModel"

filenames = Dict()
filenames["training"] = Dict(
    "tide" => joinpath("models", tide_model, "training_tides.jld2"),
    "surge" => joinpath("models", surge_model, "training_surge.jld2"),
    "waterlevel" => joinpath("models", surge_model, "training_residual.jld2")
)
filenames["validation"] = Dict(
    "tide" => joinpath("models", tide_model, "testing_tides.jld2"),
    "surge" => joinpath("models", surge_model, "testing_surge.jld2"),
    "waterlevel" => joinpath("models", surge_model, "testing_residual.jld2")
)
filenames["testing"] = Dict(
    "tide" => joinpath("models", tide_model, "testing_tides.jld2"),
    "surge" => joinpath("models", surge_model, "testing_surge.jld2"),
    "waterlevel" => joinpath("models", surge_model, "testing_residual.jld2")
)



model_pars = Dict()
model_pars["channels"] = [128, 64, 64, 64, 64, 1]
model_pars["filter"] = 2
model_pars["stride"] = 2

learning_rate = 1e-4
weight_reg = 0.5e-5
nepochs = 50
nlags = 32

name = "TestInteractionModel_small"
save_dir = "models/$(name)"

rm(save_dir, recursive=true)
if !isdir(save_dir)
    mkpath(save_dir)
end

settings = InteractionSettings(
    model_name=name,
    model_dir=save_dir,
    nepochs=nepochs,
    learning_rate=learning_rate,
    weight_reg=weight_reg,
    use_gpu=true,
    model_pars=model_pars
)

data = Dict()
for label in labels
    @show label
    ts_tide = JLD2TimeSeries(filenames[label]["tide"], varname="waterlevel")
    ts_surge = JLD2TimeSeries(filenames[label]["surge"], varname="surge")
    ts_waterlevel = JLD2TimeSeries(filenames[label]["waterlevel"], varname="surge")

    # Surge timeseries has shorter duration since it starts only surge nlags after the tide timeseries
    # Waterlevel and surge come here from the same model so have same timeseries length
    times = get_times(ts_surge)
    # ts_waterlevel = select_timespan(ts_waterlevel, times[1], times[end])
    # ts_waterlevel = select_times_by_ids(ts_waterlevel, collect(1:length(get_times(ts_waterlevel))-16))
    ts_waterlevel = TimeSeries(get_values(ts_waterlevel), times, get_names(ts_waterlevel), get_longitudes(ts_waterlevel),
        get_latitudes(ts_waterlevel), get_quantity(ts_waterlevel), get_source(ts_waterlevel))

    data[label] = Dict(
        "tide" => ts_tide,
        "surge" => ts_surge,
        "waterlevel" => ts_waterlevel
    )
end

settings.nstations = length(get_names(data["training"]["waterlevel"]))
settings.nlags = nlags
settings.npars = 2 # one for tide, one for surge

model = create_interaction_model(settings)

train_data, train_norm_stats = prepare_train_data(data["training"]["waterlevel"], data["training"]["tide"],
                data["training"]["surge"], settings)
test_data, test_norm_stats = prepare_train_data(data["testing"]["waterlevel"], data["testing"]["tide"],
                data["testing"]["surge"], settings)

model, acc_losses, train_losses, test_losses = train_model(model, settings, train_data, test_data)

save_model(model, settings)
save_settings(settings)

plot_losses(train_losses, test_losses, settings)

# Make predictions for entire data set

val_station = "VLISSGN"
start_time = "2011-01-01T00:00:00"
end_time = "2011-01-15T00:00:00"

plot_series(model, settings, 
    data["training"]["waterlevel"], data["training"]["tide"], data["training"]["surge"], 
    "training", write_series=true, write_format="something")
plot_series(model, settings, 
    data["testing"]["waterlevel"], data["testing"]["tide"], data["testing"]["surge"],
    "testing", write_series=true)
plot_series(model, settings, 
    data["testing"]["waterlevel"], data["testing"]["tide"], data["testing"]["surge"],
    "testing_14d", timerange=[start_time, end_time])
plot_series(model, settings, 
    data["training"]["waterlevel"], data["training"]["tide"], data["training"]["surge"],
    "training_14d", timerange=["2009-01-01T00:00:00", "2009-01-15T00:00:00"])

ts_single_h = select_location_by_name(data["validation"]["waterlevel"], val_station)
ts_single_h = select_timespan(ts_single_h, start_time, end_time)

ts_single_tide = select_timespan(data["validation"]["tide"], start_time, end_time)
ts_single_surge = select_timespan(data["validation"]["surge"], start_time, end_time)

y_hat = predict(model, settings, ts_single_h, ts_single_tide, ts_single_surge)