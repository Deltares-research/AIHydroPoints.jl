cd(@__DIR__)

using Pkg
Pkg.activate(".")

using AIHydroPoints

training_file="DCSM-FM_0_5nm_2008_3yr_5stations_his.jld2"
testing_file="DCSM-FM_0_5nm_2011_5stations_his.jld2"

# Use default options for everything
# Or overrise using kwargs
model_pars = Dict()
model_pars["nlayers"] = 1
model_pars["n1_feats"] = 32
model_pars["n2_feats"] = 32

learning_rate = 1.0e-3
nepochs = 100

name = "TestTideModel"
save_dir = "models/$(name)"

rm(save_dir, recursive=true)
if !isdir(save_dir)
    mkpath(save_dir)
end

settings = TideSettings(model_name=name, nepochs=nepochs, learning_rate=learning_rate, model_dir=save_dir,use_gpu=true, model_pars=model_pars)


# Load and prepare data
train_series = JLD2TimeSeries(training_file)
test_series = JLD2TimeSeries(testing_file)

settings.nstations = length(get_names(train_series))

train_data = prepare_train_data(train_series, settings)
test_data = prepare_train_data(test_series, settings)

# Define model using default function
# Or construct your own and save required hyperparameters
# for constructing the model in settings.model_pars as a Dict
# settings.model_pars = Dict(....)
model = create_tide_model(settings)

# Train model
model, acc_losses, train_losses, test_losses = train_model(model, settings, train_data, test_data)

save_model(model, settings)
save_settings(settings)

plot_losses(train_losses, test_losses, settings)

# Make predictions for entire data set

val_station = "VLISSGN"
start_time = "2011-01-01T00:00:00"
end_time = "2011-01-15T00:00:00"

plot_series(model, settings, train_series, "training", write_series=true)
plot_series(model, settings, test_series, "testing", write_series=true)
plot_series(model, settings, test_series, "testing_14d", timerange=[start_time, end_time])
plot_series(model, settings, train_series, "training_14d", timerange=["2009-01-01T00:00:00", "2009-01-15T00:00:00"])

# Or get the predicted time series itself for a single station

ts_single = select_location_by_name(test_series, val_station)
ts_single = select_timespan(ts_single, start_time, end_time)

y_hat = predict(model, settings, ts_single)