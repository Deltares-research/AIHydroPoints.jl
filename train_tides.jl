cd(@__DIR__)

using Pkg
Pkg.activate(".")

using AIHydroPoints
using Statistics
using ParameterSchedulers

labels=["training","validation","testing"]
tide_model = "TestTideModel"
filenames=Dict()

data_dir = joinpath(@__DIR__, "test_data")
filenames["training"] = Dict(
    "waterlevel" => joinpath(data_dir, "DCSM-FM_0_5nm_2008_3yr_5stations_his.jld2")
)
filenames["testing"] = Dict(
    "waterlevel" => joinpath(data_dir, "DCSM-FM_0_5nm_2011_5stations_his.jld2")
)
filenames["validation"] = Dict(
    "waterlevel" => joinpath(data_dir, "DCSM-FM_0_5nm_2011_5stations_his.jld2")
)


name = "TestTideModel"
save_dir = "models/$(name)"

rm(save_dir, recursive=true, force=true)
mkpath(save_dir)

learning_rate = 1.0e-3
lr_decay_factor = 0.9
lr_decay_rate = 50
nepochs = 2 # TESTING oNLY, CHANGE TO 250 FOR REAL TRAINING
# nepochs = 250
patience = 10
checkpoints = [40, 80, 120, 160]
val_range = ["2011-01-01T00:00:00", "2011-01-15T00:00:00"]

model_pars = Dict()
model_pars["nlayers_branch"] = 2
model_pars["nhidden_branch"] = 16
model_pars["nlayers_trunk"] = 0
model_pars["nhidden_trunk"] = 8
model_pars["nlayers_down"] = 1


settings = TideSettings(
    model_name = name,
    model_dir  = save_dir,
    use_gpu    = true,
    model_pars = model_pars,
)

train_settings = TrainingSettings(
    nepochs         = nepochs,
    checkpoints     = checkpoints,
    val_daterange   = val_range,
    learning_rate   = learning_rate,
    lr_decay_factor = lr_decay_factor,
    lr_decay_rate   = lr_decay_rate,
    patience        = patience,
)

model = TideModel(settings)

# Load and prepare data
data = Dict()
for label in labels
    @show label
    ts_h = JLD2TimeSeries(filenames[label]["waterlevel"], varname="waterlevel")
    data[label] = Dict(
        "waterlevel" => ts_h
    )
end

settings.nstations = length(get_names(data["training"]["waterlevel"]))


# Define model using default function
# Or construct your own and save required hyperparameters
# for constructing the model in settings.model_pars as a Dict
# settings.model_pars = Dict(....)

# %%


# Train model
model, acc_losses, train_losses, test_losses = train_model(model, settings, train_settings, data["training"], data["testing"])

save_model(model, settings)
save_settings(settings, train_settings)

plot_losses(train_losses, test_losses, settings, train_settings)

# Make predictions for entire data set

plot_series(model, settings, data["training"], "training", write_series=true, show_fft=true)
plot_series(model, settings, data["testing"], "testing", write_series=true, show_fft=true)
plot_series(model, settings, data["testing"], "testing_14d", timerange=train_settings.val_daterange)
plot_series(model, settings, data["training"], "training_14d", timerange=["2009-01-01T00:00:00", "2009-01-15T00:00:00"])

# Or get the predicted time series itself for a single station

# test_series = data["testing"]["waterlevel"]
# ts_single = select_location_by_name(test_series, val_station)
# ts_single = select_timespan(ts_single, start_time, end_time)

# y_hat = predict(model, settings, ts_single)