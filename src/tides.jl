using Flux
using CUDA
using ProgressMeter: Progress, next!
using Statistics

"""
    struct TideSettings

Struct that stores parameters for creating and training a model for tides.

# Arguments

- `nepochs`: Number of epochs used during training.
    (**Default**: `100`)
- `nbatches`: Number of batches to split the training datat into.
    (**Default**: `1024`)
- `learning_rate`: Learning rate of the Adam optimizer used.
    (**Default**: `1.0e-3`)
- `weight_reg`: Weight Decay parameter
    (**Default**: `1.0-e4`)
- `use_gpu`: Whether to train on gpu
    (**Default**: `false`)
- `nstation`: Number of stations used for training. Is deduced from training data when prepared, otherwise `nothing` to throw errors.
    (**Default**: `nothing`)
- `freqs`: Named tidal constituents used for training.
    (**Default**: `["SSA","K1","O1","Q1","P1","M2","S2","N2","K2","H"]`)
- `model_pars`: Dict of model parameters used to construct the tide model. The Default is set to work with the `create_tide_model` function.
    (**Default**: `Dict("nlayers=>1, "n1_feats"=>64, "n2_feats"=>64)`)
"""
@kwdef mutable struct TideSettings <: AbstractModelSettings
    model_name = "MyTideModel"
    model_dir = "MyTideModel"
    nepochs = 100
    nbatches = 1024
    learning_rate = 1.0e-3
    weight_reg = 1.0e-4
    use_gpu = false
    nstations = nothing
    freqs = ["SSA","K1","O1","Q1","P1","M2","S2","N2","K2","H"]
    model_pars = Dict(
        "nlayers" => 1,
        "n1_feats" => 64,
        "n2_feats" => 64,
    )
end

"""
    prepare_tide_data!(ts::TimeSeries, settings::TideSettings)

Prepare training data for tide model from a TimeSeries.
Returns a vector of onehot encoded input stations and doodson numbers as training input,
and waterlevels as training targets.
Also sets `nstations` in the `settings` struct.

# Arguments

- `ts::TimeSeries`: TimeSeries containing the training data.
- `settings::TideSettings`: Settings used here for the tidal frequencies used during training.
"""
function prepare_train_data(ts::TimeSeries, settings::TideSettings)
   
    times = get_times(ts)
    waterlevel = get_values(ts)

    nstations = settings.nstations

    station_index = collect(1:nstations)

    x_station, x_doodson = prepare_inputs(settings, station_index, times)
    y_waterlevel = reshape(waterlevel, 1, :)

    return (x_station, x_doodson, y_waterlevel)
end

function prepare_inputs(settings::TideSettings, station_index, times)
    nstations = settings.nstations
    freqs = settings.freqs
    ntimes = length(times)

    station_arr = station_index*ones(ntimes)'
    x_station = Flux.onehotbatch(station_arr[:], 1:nstations)

    all_times = [time for i in station_index, time in times]

    frequencies = primary_frequencies_as_doodson(freqs)
    doodson = (get_doodson_eqvals(all_times[:])*frequencies)'
    x_doodson = Float32.(vcat(cos.(doodson), sin.(doodson)))

    return x_station, x_doodson
end

####################
# Custom Input layer
####################

struct TideInputLayer{T} 
    station_params1::T
    doodson_params1::T
    station_params2::T
    doodson_params2::T
end

# Constructor
TideInputLayer(nstations, nfreqs, nfeats) = TideInputLayer(
    Dense(nstations => nfeats, identity; bias=false),
    Dense((2*nfreqs) => nfeats, identity; bias=false),
    Dense(nstations => nfeats, identity; bias=false),
    Dense((2*nfreqs) => nfeats, identity; bias=false)
)

# Forward pass
function (l::TideInputLayer)(x)
    x_station, x_doodson = x
    s1 = l.station_params1(x_station)
    d1 = l.doodson_params1(x_doodson)
    z1 = s1 .* d1
    # s2 = l.station_params2(x_station)
    # d2 = l.doodson_params2(x_doodson)
    # x2 = 0.1f0 .* s2 .* d2
    return (z1, z1)
end

Flux.@layer TideInputLayer

###################
# Custom Tide layer
###################

struct TideLayer{T} 
    direct::T
    for_product::T
end

# Constructor
TideLayer(n1_in, n1_out, n2_in, n2_out; activation=relu) = TideLayer(
    Dense(n1_in => n1_out, activation),
    Dense(n2_in => n2_out, activation)
)

# Forward pass
function (l::TideLayer)(x)
    x1, x2 = x
    # r1 = l.direct(x1)
    r2 = l.for_product(x2) .* x1
    # return (r1, r1.*r2)
    return x1 .+ r2, r2
end

Flux.@layer TideLayer

"""
    create_tide_model(settings::TideSettings) -> Return type

Description of the function

# Arguments

- `settings::TideSettings`: Argument description
"""
function create_tide_model(settings::TideSettings)
    nstations = settings.nstations
    nfreqs = length(settings.freqs)
    nlayers = settings.model_pars["nlayers"]
    n1_feats = settings.model_pars["n1_feats"]
    n2_feats = settings.model_pars["n2_feats"]

    return Chain(
            TideInputLayer(nstations, nfreqs, n1_feats),
            [TideLayer(n1_feats, n2_feats, n1_feats, n2_feats) for _ in 1:nlayers]...,
            x->sum(x[1], dims=1)./n2_feats
        )
        
end

##########
# Training
##########

function compute_loss(model, settings::TideSettings, data)
    x_station, x_doodson, y = data
    y_hat = model(x_station, x_doodson)
    return sqrt(Flux.mse(y_hat, y))
end

function train_epoch!(model, settings::TideSettings, dataloader, opt_state)
    acc_loss = 0.0f0
    for (x_station, x_doodson, y) in dataloader
        dloss, grad = Flux.withgradient(model) do m
            y_hat = m(x_station, x_doodson)
            Flux.mse(y_hat, y)
        end
        Flux.update!(opt_state, model, grad[1])
        acc_loss += dloss
    end
    return acc_loss
end

function predict(model, settings::TideSettings, ts::TimeSeries)
    times = get_times(ts)
    stations = get_names(ts)

    station_ids = 1:length(stations)
    (x_station, x_doodson) = prepare_inputs(settings, station_ids, times)
    y_hat = model(x_station, x_doodson)
    return reshape(y_hat, length(stations), length(times))
end

function plot_series(model, settings::TideSettings, ts::TimeSeries, series_name; 
    timerange::Union{Vector{DateTime}, Vector{String}, Nothing}=nothing,
    station_names::Union{Vector{String}, Nothing}=nothing, 
    write_series=false)

    
    if !isnothing(station_names)
        ts = select_locations_by_names(ts, station_names)
    end
    
    if !isnothing(timerange)
        ts = select_timespan(ts, timerange[1], timerange[2])
    end
    
    stations = get_names(ts)
    waterlevel = get_values(ts)
    times = get_times(ts)

    prediction = predict(model, settings, ts)
    errors = waterlevel .- prediction
    rmses = sqrt.(mean(abs2, errors; dims=2))

    for (ind, station) in enumerate(stations)
        h = waterlevel[ind,:]
        h_hat = prediction[ind,:]
        err = errors[ind,:]
        rmse = rmses[ind]
        p1 = plot(times, h, label="Measured", xlabel="Time", ylabel="Waterlevel", title="Station $(station) RMSE=$(rmse)")
        p2 = plot(times, h_hat, label="Predicted")
        p3 = plot(times, err, label="Residual")
        plot(p1,p2,p3,layout=(3,1))
        savefig(joinpath(settings.model_dir, "$(station)_$(series_name).png"))
    end

    if write_series
        fn_pred = joinpath(settings.model_dir, "$(series_name)_tides.nc")
        fn_res = joinpath(settings.model_dir, "$(series_name)_surge.nc")
        station_x = Float64.(get_longitudes(ts))
        station_y = Float64.(get_latitudes(ts))

        waterlevel_series_to_netcdf(fn_pred, times, prediction, stations, station_x, station_y)
        waterlevel_series_to_netcdf(fn_res, times, errors, stations, station_x, station_y)
    end

end
