using Flux
using CUDA
using Statistics
using IterTools
using JLD2
using Dates

"""
    struct SurgeSettings

Struct that stores parameters for creating and training a model for surges.

# Arguments

- `model_name`: Name of the model.
    (**Default**: `MySurgeModel`)
- `model_dir`: Path to directory where files generated during the run will be saved.
    (**Default**: `MySurgeModel`)
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
- `nstation`: Number of waterlevel stations used for training. Is deduced from training data when prepared, otherwise `nothing` to throw errors.
    (**Default**: `nothing`)
- `nwind`: Number of wind stations used for training. Is deduced from training data when prepared, otherwise `nothing` to throw errors.
    (**Default**: `nothing`)
- `nlags`: Number of previous timesteps used as input in the model.
    (*Default**: `16`)
- `model_pars`: Dict of model parameters used to construct the surge model. The Default is set to work with the `create_surge_model` function.
    (**Default**: `Dict("channels=>[32,32,32,1], "filter"=>2, "stride"=>2)`)
"""
@kwdef mutable struct SurgeSettings <: AbstractModelSettings
    model_name = "MySurgeModel"
    model_dir = "MySurgeModel"
    nepochs = 100
    nbatches = 1024
    checkpoints = nothing
    val_daterange = nothing
    learning_rate = 1.0e-3
    lr_decay_factor = nothing
    lr_decay_rate = nothing
    weight_reg = 1.0e-4
    patience = 5
    use_gpu = false
    nstations = nothing # Set per training run from train data
    nwind = nothing # Set per training run from train data
    nlags = 16
    model_pars = Dict(
        "channels" => [32,32,32,1],
        "filter" => 2,
        "stride" => 2
    )
end

###################
# Input Preparation
###################


"""
    prepare_train_data(ts_waterlevel::TimeSeries, ts_wind_x::TimeSeries, ts_wind_y::TimeSeries, ts_press::TimeSeries, settings::SurgeSettings)

Prepare training data for surge model from waterlevel, wind stress, and pressure TimeSeries.
Returns a vector of onehot encoded input stations and wind stress and pressure at previous timesteps as input,
and waterlevel (surge) as training targets.

# Arguments

- `ts_waterlevel::TimeSeries`: TimeSeries containing waterlevel (surge) data.
- `ts_wind_x::TimeSeries`: TimeSeries containing wind stress (x-direction) data.
- `ts_wind_y::TimeSeries`: TimeSeries containing wind stress (y-direction) data.
- `ts_press::TimeSeries`: TimeSeries containing pressure data.
- `settings::SurgeSettings`: Surge model settings.
"""
function prepare_train_data(data_dict::Dict{String, <:AbstractTimeSeries}, settings::SurgeSettings)

    ts_waterlevel = data_dict["waterlevel"]
    ts_wind_x = data_dict["wind_x"]
    ts_wind_y = data_dict["wind_y"]
    ts_press = data_dict["pressure"]

    times = get_times(ts_waterlevel)

    waterlevel = get_values(ts_waterlevel)
    stress_x = get_values(ts_wind_x)
    stress_y = get_values(ts_wind_y)
    press = get_values(ts_press)

    lats = get_latitudes(ts_waterlevel)
    lons = get_longitudes(ts_waterlevel)

    nlags = settings.nlags
    nstations = settings.nstations
    station_index = 1:nstations

    x_station, x_stress_press = prepare_inputs(settings, station_index, times, stress_x, stress_y, press, lats, lons)
    
    time_idx = nlags:length(times)
    y_waterlevel = zeros(Float32, nstations, nlags, length(time_idx))
    for itime in time_idx

        waterlevel_block = Float32.(waterlevel[:,itime-nlags+1:itime])
        y_waterlevel[:,:,itime-nlags+1] = waterlevel_block

    end
    
    # y_waterlevel = waterlevel[:, nlags:end]
    # y_waterlevel = reshape(waterlevel[:, nlags:end], 1, :)
    return x_station, x_stress_press, y_waterlevel
end

"""
    prepare_inputs(settings::SurgeSettings, station_index, times, stress_x, stress_y, press)

Create surge model inputs based on station indices, time, wind stress, pressure.

# Arguments

- `settings::SurgeSettings`: Surge model settings
- `station_index`: Indices of waterlevel stations to prep data for.
- `times`: Input times.
- `stress_x`: Wind stress (x-direction)
- `stress_y`: Wind stress (y-direction)
- `press`: Pressure
"""
function prepare_inputs(settings::SurgeSettings, station_index, times, stress_x, stress_y, press, h_lats, h_lons)
    nlags = settings.nlags
    time_idx = nlags:length(times)
    ntimes = length(time_idx)
    nstations = settings.nstations
    nwind = settings.nwind

    press = 2e-4*(press.-1e5)

    dayperiod = 365.25

    # Onehot encoding of station index
    # station_arr = station_index*ones(ntimes)'
    # x_station = station_arr
    # x_station = Flux.onehotbatch(station_arr[:], 1:nstations)

    x_station = zeros(Float32, 6, nstations, ntimes)

    times_day = Dates.dayofyear.(times[time_idx])
    times_cos = cos.(2π .*times_day./dayperiod)
    times_sin = sin.(2π .*times_day./dayperiod)

    lats_cos = cos.(deg2rad.(h_lats))
    lats_sin = sin.(deg2rad.(h_lats))
    lons_cos = cos.(deg2rad.(h_lons))
    lons_sin = sin.(deg2rad.(h_lons))

    x_station[1,:,:] .= Float32.(lats_cos)
    x_station[2,:,:] .= Float32.(lats_sin)
    x_station[3,:,:] .= Float32.(lons_cos)
    x_station[4,:,:] .= Float32.(lons_sin)
    x_station[5,:,:] .= Float32.(times_cos)'
    x_station[6,:,:] .= Float32.(times_sin)'



    # all training times for all stations
    all_times = [itime for i in 1:length(station_index), itime in time_idx][:]

    # Create input stress, press data
    x_stress_press = zeros(Float32, 3*nwind, nlags, ntimes)
    for itime in time_idx
        stress_x_block = stress_x[:, itime-nlags+1:itime]
        stress_y_block = stress_y[:, itime-nlags+1:itime]
        press_block = press[:, itime-nlags+1:itime]
        x_block = Float32.(vcat(stress_x_block, stress_y_block, press_block))
        x_stress_press[:,:,itime-nlags+1] .= x_block
        # x_block = Float32.(permutedims(cat(stress_x_block, stress_y_block, press_block, dims=3), [2,1,3]))
        # x_stress_press[:,:,:,itime-nlags+1] .= x_block
        # for istation in 1:length(station_index) # need copy of x_block for each station
        #     isample = (itime-nlags)*(length(station_index))+istation
        #     x_stress_press[:,:,:,isample] .= x_block
        # end
    end

    return x_station, x_stress_press
end

####################
# Custom Input Layer
####################

"""
    struct WindInputLayer{T}

Input layer to surge model encoding the station indices
"""
struct WindInputLayer{T}
    station_params::T
end

# Constructor
"""
    WindInputLayer(nstations, nlags, npars)

WindInputLayer constructor using number of stations, number of previous timesteps,
and number of input paramters.

# Arguments

- `nstations`: Number of input waterlevel stations
- `nlags`: Number of previous timesteps seen by model
- `npars`: Total number of input parameters (=features) of the input layer
"""
WindInputLayer(nstations, nlags, npars) = WindInputLayer(
    Dense(nstations => (nlags*npars), identity; bias=false)
)

# Forward pass
function (l::WindInputLayer)(x)
    x_station, x_wind = x
    nlags, npars, nbatch = size(x_wind)

    s1 = l.station_params(x_station)
    s1 = reshape(s1, (nlags, npars, nbatch))
    z1 = s1 .* x_wind
    return z1
end

Flux.@layer WindInputLayer

"""
    create_surge_model(settings::SurgeSettings) 

Create a default Surge Model based on model parameters stored in `settings.model_pars`,
using WindInputLayer and Conv layers.

# Arguments

- `settings::SurgeSettings`: Surge model settings
"""
function create_surge_model(settings::SurgeSettings)
    nstations = settings.nstations
    nlags = settings.nlags
    nchannels = settings.model_pars["channels"]
    npars = 3*settings.nwind
    filtersize = settings.model_pars["filter"]
    strides = settings.model_pars["stride"]

    channels = [npars, nchannels...]

    return Chain(
        WindInputLayer(nstations, nlags, npars),
        [Conv((filtersize,), in => out, relu, stride=(strides,), pad=SamePad()) for (in, out) in partition(channels[1:end-1], 2, 1)]...,
        Conv((filtersize,), channels[end-1] => channels[end], identity, stride=(strides,), pad=SamePad()),
        Flux.flatten,
    ) 
end

struct SurgeModel{P, Q, R, T} 
    branch_net::P
    trunk_net::Q
    downsample::R
    adjacency::AbstractArray{T,2}
end

function SurgeModel(gn, nlags, nwind, nembed, theta, nheads, nlayers_branch, nlayers_trunk, nhidden_trunk)
    # For 3 pars per wind station, and nlags previous timesteps
    embed = Embedder(3*nwind, nembed)
    deembed = Deembedder(embed)
    pos_embed = SinCosPosEmbedder(nembed, nlags, theta=theta)

    branch_net = Chain(embed, 
        pos_embed, 
        [Transformer(nembed, nheads) for _ in 1:nlayers_branch]...,
        deembed,
        x -> reshape(x, (nwind, 3, nlags, :))
    )

    trunk_net = Chain(
        Dense(6=>nhidden_trunk),
        [Dense(nhidden_trunk => nhidden_trunk) for _ in 1:nlayers_trunk]...,
        Dense(nhidden_trunk => nwind)
    )

    # down = Chain(
    #     x->permutedims(x, (2,1,3)),
    #     MaxPool((3,)),
    #     x->permutedims(x, (2,1,3))
    # )
    down = Conv((1,), 3*nlags=>nlags, identity, stride=(1,), pad=SamePad())

    return SurgeModel(branch_net, trunk_net, down, gn.adjacency)
end

function SurgeModel(gn::GraphNetwork, settings::SurgeSettings)
    nlags = settings.nlags
    nwind = settings.nwind
    nembed = settings.model_pars["nembed"]
    theta = settings.model_pars["theta"]
    nheads = settings.model_pars["nheads"]
    nlayers_branch = settings.model_pars["nlayers_branch"]
    nlayers_trunk = settings.model_pars["nlayers_trunk"]
    nhidden_trunk = settings.model_pars["nhidden_trunk"]

    return SurgeModel(gn, nlags, nwind, nembed, theta, nheads, nlayers_branch, nlayers_trunk, nhidden_trunk)
    
end

function (m::SurgeModel)(x_station, x_wind)
    nbatch = size(x_station)[end]

    branch_out = m.branch_net(x_wind)
    trunk_out = m.trunk_net(x_station)

    nwind = size(branch_out,1)

    merged = batched_mul(batched_transpose(trunk_out.*m.adjacency),
        reshape(branch_out, (nwind, :, nbatch)))
    return m.downsample(merged)
end

@Flux.layer SurgeModel

##########
# Training
##########

function compute_loss(model, settings::SurgeSettings, data)
    x_station, x_stress_press, y = data
    y_hat = model(x_station, x_stress_press)
    return sqrt(Flux.mse(y_hat, y))
end

function train_epoch!(model, settings::SurgeSettings, dataloader, opt_state)
    acc_loss = 0.0f0
    for (x_station, x_stress_press, y) in dataloader
        dloss, grads = Flux.withgradient(model) do m
            y_hat = m(x_station, x_stress_press)
            Flux.mse(y_hat, y)
        end
        Flux.update!(opt_state, model, grads[1])
        acc_loss += dloss
    end
    return acc_loss
end

function predict(model, settings::SurgeSettings, ts_h::TimeSeries, ts_wind_x::TimeSeries, ts_wind_y::TimeSeries, ts_press::TimeSeries)
    times = get_times(ts_wind_x)
    nstations = length(get_names(ts_h))
    nlags = settings.nlags
    station_idx = 1:nstations

    stress_x = get_values(ts_wind_x)
    stress_y = get_values(ts_wind_y)
    press = get_values(ts_press)

    lats = get_latitudes(ts_h)
    lons = get_longitudes(ts_h)

    x_station, x_stress_press = prepare_inputs(settings, station_idx, times, stress_x, stress_y, press, lats, lons)
    y_hat = model(x_station, x_stress_press)
    
    return y_hat[:, settings.nlags, :]
    # return reshape(y_hat, nstations, length(times)-nlags+1)
end

function plot_series(model, settings::SurgeSettings, data_dict::Dict{String, <:AbstractTimeSeries}, series_name;
    timerange::Union{Vector{DateTime}, Vector{String}, Nothing}=nothing,
    station_names::Union{Vector{String}, Nothing}=nothing,
    write_series=false, write_format="jld2")

    ts_h = data_dict["waterlevel"]
    ts_wind_x = data_dict["wind_x"]
    ts_wind_y = data_dict["wind_y"]
    ts_press = data_dict["pressure"]
    
    if !isnothing(station_names)
        ts_h = select_locations_by_names(ts_h, station_names)
    end

    if !isnothing(timerange)
        ts_h = select_timespan(ts_h, timerange[1], timerange[2])
        ts_wind_x = select_timespan(ts_wind_x, timerange[1], timerange[2])
        ts_wind_y = select_timespan(ts_wind_y, timerange[1], timerange[2])
        ts_press = select_timespan(ts_press, timerange[1], timerange[2])
    end

    nlags = settings.nlags
    stations = get_names(ts_h)
    waterlevel = get_values(ts_h)[:, nlags:end]
    times = get_times(ts_h)[nlags:end]

    prediction = predict(model, settings, ts_h, ts_wind_x, ts_wind_y, ts_press)
    errors = waterlevel .- prediction
    rmses = sqrt.(mean(abs2, errors; dims=2))

    for (ind, station) in enumerate(stations)
        h = waterlevel[ind,:]
        h_hat = prediction[ind,:]
        err = errors[ind,:]
        rmse = rmses[ind]

        p1 = plot(times, h, label="Ground Truth", xlabel="Time", ylabel="Waterlevel", title="Station $station RMSE=$rmse")
        plot!(p1, times, h_hat, label="Predicted")
        p2 = plot(times, err, label="Residual")

        plot(p1,p2, layout=(2,1))
        savefig(joinpath(settings.model_dir, "$(station)_$(series_name).png"))
    end

    if write_series
        fn_pred = joinpath(settings.model_dir, "$(series_name)_surge")
        fn_res = joinpath(settings.model_dir, "$(series_name)_residual")
        station_x = Float64.(get_longitudes(ts_h))
        station_y = Float64.(get_latitudes(ts_h))

        if write_format == "netcdf"
            write_to_netcdf(TimeSeries(Float32.(prediction), times, stations, station_x, station_y, "surge",    get_source(ts_h)), fn_pred*".nc")
            write_to_netcdf(TimeSeries(Float32.(errors),     times, stations, station_x, station_y, "residual", get_source(ts_h)), fn_res*".nc")

        else
            if write_format != "jld2"
                @warn "Unknown writing format $(write_format), using defaulft format JLD2."
            end 
            ext = ".jld2"
            save(fn_pred*ext,
                Dict(
                    "station_x_coordinate" => station_x,
                    "station_y_coordinate" => station_y,
                    "station_names" => stations,
                    "times" => times,
                    "surge" => prediction
                )    
            )
            save(fn_res*ext,
                Dict(
                    "station_x_coordinate" => station_x,
                    "station_y_coordinate" => station_y,
                    "station_names" => stations,
                    "times" => times,
                    "surge" => errors
                )    
            )            
        end
    end
end