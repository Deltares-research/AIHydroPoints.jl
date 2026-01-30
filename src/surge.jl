using Flux
using CUDA
using Statistics
using IterTools
using JLD2

@kwdef mutable struct SurgeSettings <: AbstractModelSettings
    model_name = "MySurgeModel"
    model_dir = "MySurgeModel"
    nepochs = 100
    nbatches = 1024
    learning_rate = 1.0e-3
    weight_reg = 1.0e-4
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

function prepare_train_data(ts_waterlevel::TimeSeries, ts_wind_x::TimeSeries, ts_wind_y::TimeSeries, ts_press::TimeSeries, settings::SurgeSettings)
    times = get_times(ts_waterlevel)

    waterlevel = get_values(ts_waterlevel)
    stress_x = get_values(ts_wind_x)
    stress_y = get_values(ts_wind_y)
    press = get_values(ts_press)

    nlags = settings.nlags
    nstations = settings.nstations
    station_index = 1:nstations

    x_station, x_stress_press = prepare_inputs(settings, station_index, times, stress_x, stress_y, press)
    y_waterlevel = reshape(waterlevel[:, nlags:end], 1, :)
    return x_station, x_stress_press, y_waterlevel
end

function prepare_inputs(settings::SurgeSettings, station_index, times, stress_x, stress_y, press)
    nlags = settings.nlags
    time_idx = nlags:length(times)
    ntimes = length(time_idx)
    nstations = settings.nstations
    nwind = settings.nwind

    press = 2e-4*(press.-1e5)

    # Onehot encoding of station index
    station_arr = station_index*ones(ntimes)'
    x_station = Flux.onehotbatch(station_arr[:], 1:nstations)

    # all training times for all stations
    all_times = [itime for i in 1:length(station_index), itime in time_idx][:]

    # Create input stress, press data
    x_stress_press = zeros(Float32, nlags, nwind*3, length(station_index)*ntimes)
    for itime in time_idx
        stress_x_block = stress_x[:, itime-nlags+1:itime]
        stress_y_block = stress_y[:, itime-nlags+1:itime]
        press_block = press[:, itime-nlags+1:itime]
        x_block = Float32.(vcat(stress_x_block, stress_y_block, press_block))'
        for istation in 1:length(station_index) # need copy of x_block for each station
            isample = (itime-nlags)*(length(station_index))+istation
            x_stress_press[:,:,isample] .= x_block
        end
    end

    return x_station, x_stress_press
end

####################
# Custom Input Layer
####################

struct WindInputLayer{T}
    station_params::T
end

# Constructor
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

    x_station, x_stress_press = prepare_inputs(settings, station_idx, times, stress_x, stress_y, press)
    y_hat = model(x_station, x_stress_press)
    
    return reshape(y_hat, nstations, length(times)-nlags+1)
end

function plot_series(model, settings::SurgeSettings, ts_h::TimeSeries,
    ts_wind_x::TimeSeries, ts_wind_y::TimeSeries, ts_press::TimeSeries, series_name;
    timerange::Union{Vector{DateTime}, Vector{String}, Nothing}=nothing,
    station_names::Union{Vector{String}, Nothing}=nothing,
    write_series=false, write_format="jld2")
    
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
            ext = ".nc"
            waterlevel_series_to_netcdf(fn_pred*ext, times, prediction, stations, station_x, station_y)
            waterlevel_series_to_netcdf(fn_res*ext, times, errors, stations, station_x, station_y)

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