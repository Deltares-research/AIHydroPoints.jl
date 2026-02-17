using CUDA
using Statistics
using JLD2
using IterTools

@kwdef mutable struct InteractionSettings <: AbstractModelSettings
    model_name = "MyInteractionModel"
    model_dir = "MyInteractionModel"
    nepochs = 100
    nbatches = 1024
    patience = 5
    learning_rate = 1.0e-3
    weight_reg = 1.0e-4
    use_gpu = false
    nstations = nothing # Set per training run from train data
    npars = nothing # Set per training run from train data
    nlags = 16
    model_pars = Dict(
        "channels" => [128,64,64,64,32,1],
        "filter" => 2,
        "stride" => 2,
    )
end

###################
# Input Preparation
###################

function prepare_train_data(ts_waterlevel::TimeSeries, ts_tides::TimeSeries, ts_surge::TimeSeries, settings::InteractionSettings)
    # times = get_times(ts_waterlevel)[16:end]
    times = get_times(ts_waterlevel)

    waterlevel = get_values(ts_waterlevel)
    tides = get_values(ts_tides)
    surge = get_values(ts_surge)

    nlags = settings.nlags
    nstations = settings.nstations
    station_index = 1:nstations

    (x_station, x_tide_surge), stats_tidesurge = prepare_inputs(settings, station_index, times, tides, surge)
    y_waterlevel = reshape(waterlevel[:, nlags:end], 1, :)

    mu_waterlevel = mean(y_waterlevel[:])
    std_waterlevel = std(y_waterlevel[:])

    y_waterlevel = (y_waterlevel .- mu_waterlevel)./std_waterlevel

    return (x_station, x_tide_surge, y_waterlevel), (mu_waterlevel, std_waterlevel, stats_tidesurge...)    
end

function prepare_inputs(settings::InteractionSettings, station_index, times, tides, surge)
    nlags = settings.nlags
    time_idx = nlags:length(times)
    ntimes = length(time_idx)
    nstations = settings.nstations

    station_arr = station_index*ones(ntimes)'
    x_station = Flux.onehotbatch(station_arr[:], 1:nstations)

    x_tide_surge = zeros(Float32, nlags, settings.npars, length(station_index)*ntimes)
    for itime in time_idx
        surge_block = surge[:, itime-nlags+1:itime]
        tide_block = tides[:, itime-nlags+1:itime]
        x_block = Float32.(vcat(surge_block, tide_block))'
        for istation in 1:length(station_index)
            isample = (itime-nlags)*length(station_index)+istation
            x_tide_surge[:,:,isample].=x_block[:, istation:istation+1]
        end
    end

    mu_tidesurge = mean(x_tide_surge[:])
    std_tidesurge = std(x_tide_surge[:])

    x_tide_surge = (x_tide_surge .- mu_tidesurge)./std_tidesurge

    return (x_station, x_tide_surge), (mu_tidesurge, std_tidesurge)
end

####################
# Custom Input Layer
####################

struct InteractionInputLayer{T}
    station_params::T
end

InteractionInputLayer(nstations, nlags, npars) = InteractionInputLayer(
    Dense(nstations => nlags*npars, identity; bias=false)
)

function (l::InteractionInputLayer)(x)
    x_station, x_tide_surge = x
    nlags, npars, nbatch = size(x_tide_surge)
    
    s1 = l.station_params(x_station)
    s1 = reshape(s1, (nlags, npars, nbatch))

    z1 = s1 .* x_tide_surge
    return z1
end

Flux.@layer InteractionInputLayer

function create_interaction_model(settings::InteractionSettings)
    nstations = settings.nstations
    nlags = settings.nlags
    nchannels = settings.model_pars["channels"]
    npars = settings.npars
    filtersize = settings.model_pars["filter"]
    strides = settings.model_pars["stride"]

    channels = [npars, nchannels...]

    return Chain(
        InteractionInputLayer(nstations, nlags, npars),
        [Conv((filtersize,), in => out, tanh, stride=(strides,), pad=SamePad()) for (in,out) in partition(channels[1:end-1], 2, 1)]...,
        Conv((filtersize,), channels[end-1] => channels[end], identity, stride=(strides,), pad=SamePad()),
        Flux.flatten,
    )

end

##########
# Training
##########

function compute_loss(model, settings::InteractionSettings, data)
    x_station, x_tide_surge, y = data
    y_hat = model(x_station, x_tide_surge)
    return sqrt(Flux.mse(y_hat, y))
end

function train_epoch!(model, settings::InteractionSettings, dataloader, opt_state)
    acc_loss = 0.0f0
    for (x_station, x_tide_surge, y) in dataloader
        dloss, grads = Flux.withgradient(model) do m
            y_hat = m(x_station, x_tide_surge)
            Flux.mse(y_hat, y)
        end
        Flux.update!(opt_state, model, grads[1])
        acc_loss += dloss
    end
    return acc_loss
end

function predict(model, settings::InteractionSettings, ts_waterlevel::TimeSeries, ts_tides::TimeSeries, ts_surge::TimeSeries)
    # times = get_times(ts_waterlevel)[16:end]
    times = get_times(ts_waterlevel)
    nstations = length(get_names(ts_waterlevel))
    nlags = settings.nlags
    station_idx = 1:nstations

    tides = get_values(ts_tides)
    surge = get_values(ts_surge)

    (x_station, x_tide_surge), (mu_ts, std_ts) = prepare_inputs(settings, station_idx, times, tides, surge)
    y_hat = std_ts.*model(x_station, x_tide_surge) .+ mu_ts

    return reshape(y_hat, nstations, length(times)-nlags+1)
end

function plot_series(model, settings::InteractionSettings, ts_waterlevel::TimeSeries,
    ts_tides::TimeSeries, ts_surge::TimeSeries, series_name;
    timerange::Union{Vector{DateTime}, Vector{String}, Nothing}=nothing,
    station_names::Union{Vector{String}, Nothing}=nothing,
    write_series=false, write_format="jld2")

    if !isnothing(station_names)
        ts_waterlevel = select_locations_by_names(ts_waterlevel, station_names)
    end

    if !isnothing(timerange)
        ts_waterlevel = select_timespan(ts_waterlevel, timerange[1], timerange[2])
        ts_tides = select_timespan(ts_tides, timerange[1], timerange[2])
        ts_surge = select_timespan(ts_surge, timerange[1], timerange[2])
    end

    nlags = settings.nlags
    stations = get_names(ts_waterlevel)

    waterlevel = get_values(ts_waterlevel)[:, nlags:end]
    # times = get_times(ts_waterlevel)[15+nlags:end]
    times = get_times(ts_waterlevel)[nlags:end]

    prediction = predict(model, settings, ts_waterlevel, ts_tides, ts_surge)
    errors = waterlevel .- prediction
    rmses = sqrt.(mean(abs, errors; dims=2))

    for (ind, station) in enumerate(stations)
        h = waterlevel[ind,:]
        h_hat = prediction[ind,:]
        err = errors[ind,:]
        rmse = rmses[ind]

        p1 = plot(times, h, label="Ground Truth", xlabel="Time", ylabel="Waterlevel", title="Station $station RMSE=$rmse")
        plot!(p1, times, h_hat, label="Predicted")
        p2 = plot(times, err, label="Residual")

        plot(p1, p2, layout=(2,1))
        savefig(joinpath(settings.model_dir, "$(station)_$(series_name).png"))
    end

    if write_series
        fn_pred = joinpath(settings.model_dir, "$(series_name)_interaction")
        fn_res = joinpath(settings.model_dir, "$(series_name)_residual")

        station_x = Float64.(get_longitudes(ts_waterlevel))
        station_y = Float64.(get_latitudes(ts_waterlevel))

        if write_format == "netcdf"
            ext = ".nc"
            waterlevel_series_to_netcdf(fn_pred*ext, times, prediction, stations, station_x, station_y)
            waterlevel_series_to_netcdf(fn_res*ext, times, errors, stations, station_x, station_y)
        else
            if write_format != "jld2"
                @warn "Unknown writing format $(write_format), using default format JLD2"
            end
            ext = ".jld2"
            save(fn_pred*ext,
                Dict(
                    "station_x_coordinate" => station_x,
                    "station_y_coordinate" => station_y,
                    "station_names" => stations,
                    "times" => times,
                    "interaction" => prediction
                )
            )
            save(fn_res*ext,
                Dict(
                    "station_x_coordinate" => station_x,
                    "station_y_coordinate" => station_y,
                    "station_names" => stations,
                    "times" => times,
                    "interaction" => errors
                )
            )
        end
    end
end