cd(@__DIR__)

using Pkg
Pkg.activate(".")

using AIHydroPoints
using Statistics
using ParameterSchedulers

labels=["training","validation","testing"]
tide_model = "TestTideModel"
filenames=Dict()

filenames["training"] = Dict(
    "waterlevel" => "DCSM-FM_0_5nm_2008_3yr_5stations_his.jld2" 
)
filenames["testing"] = Dict(
    "waterlevel" => "DCSM-FM_0_5nm_2011_5stations_his.jld2" 
)
filenames["validation"] = Dict(
    "waterlevel" => "DCSM-FM_0_5nm_2011_5stations_his.jld2" 
)

nlayers_range = [1]
nfeats_range = [8]

# Use default options for everything
# Or overrise using kwargs
# model_pars = Dict()
# model_pars["nlayers"] = 1
# model_pars["n1_feats"] = 32
# model_pars["n2_feats"] = 32

# learning_rate = 1.0e-3
lrs = [1e-3]
weight_regs = [1e-4]
# lrs = [3e-3, 1e-3, 3e-4, 1e-4]
# weight_regs = [3e-4, 1e-4, 3e-5, 1e-5]
nepochs = 100

val_station = "VLISSGN"
start_time = "2011-01-01T00:00:00"
end_time = "2011-01-15T00:00:00"

name = "TestTideModel"
save_dir = "models/$(name)"

rm(save_dir, recursive=true)
if !isdir(save_dir)
    mkpath(save_dir)
end

settings = TideSettings(model_name=name, nepochs=nepochs, model_dir=save_dir,use_gpu=true)


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

settings.lr_decay_factor = 0.5
settings.lr_decay_rate = 25

# Define model using default function
# Or construct your own and save required hyperparameters
# for constructing the model in settings.model_pars as a Dict
# settings.model_pars = Dict(....)

function hyperpar_search(nlayers, nfeats, lrs, weight_regs)
    min_loss = 1e3
    min_pars = Dict()
    min_lr = 0.
    min_weight_reg = 0

    for pars in Base.product(nlayers, nfeats, lrs, weight_regs)
        nlayers = pars[1]
        n1feats = pars[2]
        n2feats = pars[2]

        lr = pars[3]
        weight_reg = pars[4]

        model_pars = Dict()
        model_pars["nlayers"] = nlayers
        model_pars["n1_feats"] = n1feats
        model_pars["n2_feats"] = n2feats

        @info "Training model with pars $model_pars, lr $lr, weight_reg $weight_reg"

        settings.model_pars = model_pars
        settings.learning_rate = lr
        settings.weight_reg = weight_reg

        model = create_tide_model(settings)

        model, acc_losses, train_losses, test_losses = train_model(model, settings, data["training"], data["testing"])

        mean_test_loss = mean(test_losses[end-5:end])

        if mean_test_loss < min_loss
            @info "New best model found with test_loss $mean_test_loss , pars $model_pars, lr $lr, weight_reg $weight_reg"
            min_loss = mean_test_loss
            min_pars = model_pars
            min_lr = lr
            min_weight_reg = weight_reg
        end

    end

    return min_loss, min_pars, min_lr, min_weight_reg

end

min_loss, min_pars, min_lr, min_weight_reg = hyperpar_search(nlayers_range, nfeats_range, lrs, weight_regs)

@info "Min loss found: $min_loss , at pars: $min_pars"

settings.model_pars = min_pars
settings.learning_rate = min_lr
settings.weight_reg = min_weight_reg
settings.checkpoints = [25,50,75]
settings.val_daterange = [start_time, end_time]

model = create_tide_model(settings)

# Train model
model, acc_losses, train_losses, test_losses = train_model(model, settings, data["training"], data["testing"])

save_model(model, settings)
save_settings(settings)

plot_losses(train_losses, test_losses, settings)

# Make predictions for entire data set

plot_series(model, settings, data["training"], "training", write_series=true)
plot_series(model, settings, data["testing"], "testing", write_series=true)
plot_series(model, settings, data["testing"], "testing_14d", timerange=[start_time, end_time])
plot_series(model, settings, data["training"], "training_14d", timerange=["2009-01-01T00:00:00", "2009-01-15T00:00:00"])

# Or get the predicted time series itself for a single station

test_series = data["testing"]["waterlevel"]
ts_single = select_location_by_name(test_series, val_station)
ts_single = select_timespan(ts_single, start_time, end_time)

y_hat = predict(model, settings, ts_single)