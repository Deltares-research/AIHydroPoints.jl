using Flux
using CUDA
using Statistics
using JLD2
using FFTW

"""
    struct TideSettings

Inference-time parameters for the tide model.
Training hyperparameters (epochs, learning rate, etc.) are stored separately in `TrainingSettings`.

# Fields

- `model_name`: Name of the model.
    (**Default**: `"MyTideModel"`)
- `model_dir`: Directory where files generated during the run will be saved.
    (**Default**: `"MyTideModel"`)
- `use_gpu`: Whether to train/run on GPU.
    (**Default**: `false`)
- `nstations`: Number of waterlevel stations. Set from training data.
    (**Default**: `nothing`)
- `freqs`: Named tidal constituents used for training.
    (**Default**: `["SSA","K1","O1","Q1","P1","M2","S2","N2","K2","H"]`)
- `model_pars`: Dict of model architecture parameters.
    (**Default**: `Dict("nlayers"=>1, "n1_feats"=>64, "n2_feats"=>64)`)
"""
@kwdef mutable struct TideSettings <: AbstractModelSettings
    model_name = "MyTideModel"
    model_dir = "MyTideModel"
    use_gpu = false
    nstations = nothing # Set per training run from train data
    freqs = ["SSA","K1","O1","Q1","P1","M2","S2","N2","K2","H"]
    model_pars = Dict(
        "nlayers" => 1,
        "n1_feats" => 64,
        "n2_feats" => 64,
    )
end

"""
    prepare_train_data(ts::TimeSeries, settings::TideSettings)

Prepare training data for tide model from a TimeSeries.
Returns a vector of onehot encoded input stations and doodson numbers as training input,
and waterlevels as training targets.

# Arguments

- `ts::TimeSeries`: TimeSeries containing the training data.
- `settings::TideSettings`: Settings used here for the tidal frequencies used during training.
"""
function prepare_train_data(data_dict::Dict{String, <:AbstractTimeSeries}, settings::TideSettings)
   
    ts = data_dict["waterlevel"]

    times = get_times(ts)
    waterlevel = get_values(ts)

    lats = get_latitudes(ts)
    lons = get_longitudes(ts)

    station_index = collect(1:settings.nstations)

    x_station, x_doodson = prepare_inputs(settings, lats, lons, times)
    # y_waterlevel = reshape(waterlevel, 1, :)
    y_waterlevel = waterlevel

    return (x_station, x_doodson, y_waterlevel)
end

"""
    prepare_inputs(settings::TideSettings, station_index, times)

Creates the tide model input arrays based on the index of the station in the data set,
and an array of times.

# Arguments

- `settings::TideSettings`: Tide model settings.
- `station_index`: Index/Indices of stations to prep data for.
- `times`: Times used as input.
"""

function prepare_inputs(settings::TideSettings, lats, lons, times)
    nstations = settings.nstations
    freqs = settings.freqs
    ntimes = length(times)

    x_station = zeros(Float32, 4, nstations, ntimes)

    lats_cos = cos.(deg2rad.(lats))
    lats_sin = sin.(deg2rad.(lats))
    lons_cos = cos.(deg2rad.(lons))
    lons_sin = sin.(deg2rad.(lons))

    x_station[1,:,:] .= Float32.(lats_cos)
    x_station[2,:,:] .= Float32.(lats_sin)
    x_station[3,:,:] .= Float32.(lons_cos)
    x_station[4,:,:] .= Float32.(lons_sin)

    # station_arr = station_index*ones(ntimes)'
    # x_station = Flux.onehotbatch(station_arr[:], 1:nstations)

    # all_times = [time for i in station_index, time in times]

    frequencies = primary_frequencies_as_doodson(freqs)
    # doodson = (get_doodson_eqvals(all_times[:])*frequencies)'
    doodson = (get_doodson_eqvals(times)*frequencies)'
    x_doodson = Float32.(vcat(cos.(doodson), sin.(doodson)))

    return x_station, x_doodson
end

####################
# Custom Input layer
####################

"""
    struct TideInputLayer{T}

Input layer to the tide model encoding the station indices and doodson pars.
"""
struct TideInputLayer{T} 
    station_params1::T
    doodson_params1::T
    station_params2::T
    doodson_params2::T
end

# Constructor
"""
    TideInputLayer(nstations, nfreqs, nfeats)

TideInputLayer constructor using the number of stations, number of tidal frequencies,
and number of hidden features in the layer.

# Arguments

- `nstations`: Number of stations in the training data
- `nfreqs`: Number of tidal frequencies used during training
- `nfeats`: Number of hidden features in the layer
"""
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

"""
    struct TideLayer{T}

Processing layer used in tide model with two branches
"""
struct TideLayer{T} 
    direct::T
    for_product::T
end

# Constructor
"""
    TideLayer(n1_in, n1_out, n2_in, n2_out; kwargs...)

Constructor function to build a TideLayer using a Dense layer in each branch.
# Arguments

- `n1_in`: Number of input features in first branch
- `n1_out`: Number of output features in first branch
- `n2_in`: Number of input features in second branch
- `n2_out`: Number of output features in second branch

# Keywords

- `activation`: Dense layer activation function
    (**Default**: `relu`)
"""
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
    create_tide_model(settings::TideSettings)

Create a default Tide Model based on hyperparameters defined in the settings
and using the TideInputLayer, TideLayer defined previously.

# Arguments

- `settings::TideSettings`: settings for the TideModel
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

struct TideModel{P, Q, T}
    branch::P
    trunk::Q
    downsample::T
end

function TideModel(nfreqs, nlayers_branch, nhidden_branch, nlayers_trunk, nhidden_trunk, nlayers_down, activ_func)
    branch = Chain(
        Dense(2*nfreqs, nhidden_branch, activ_func),
        [Dense(nhidden_branch, nhidden_branch, activ_func) for _ in 1:nlayers_branch]...,
        Dense(nhidden_branch, nhidden_branch, tanh),
        # x->(reshape(x, nhidden_trunk, 2, :))
    )

    trunk = Chain(
        Dense(4, nhidden_trunk, activ_func),
        [Dense(nhidden_trunk, nhidden_trunk, activ_func) for _ in 1:nlayers_trunk]...,
        Dense(nhidden_trunk, 2, tanh)
    )

    # down = Dense(nhidden_branch, 1)
    down  = Chain(
        [Dense(nhidden_branch, nhidden_branch, activ_func) for _ in 1:nlayers_down]...,
        Dense(nhidden_branch, 1)
    )
    # down = x->sum(x, dims=1)

    return TideModel(branch, trunk, down)
end

function TideModel(settings::TideSettings)
    nfreqs = length(settings.freqs)
    nlayers_branch = settings.model_pars["nlayers_branch"]
    nhidden_branch = settings.model_pars["nhidden_branch"]
    nlayers_trunk = settings.model_pars["nlayers_trunk"]
    nhidden_trunk = settings.model_pars["nhidden_trunk"]
    nlayers_down = settings.model_pars["nlayers_down"]

    return TideModel(nfreqs, nlayers_branch, nhidden_branch, nlayers_trunk, nhidden_trunk, nlayers_down, leakyrelu)
end

function (m::TideModel)(x_stations, x_doodson)
    branch_out = m.branch(x_doodson)
    trunk_out = m.trunk(x_stations)

    # merged = cat([trunk_out[2,:,:].*slice .+ trunk_out[2,:,:] for slice in eachslice(trunk_out, dims=2)]..., dims=3)
    merged = cat([slice[1,:]'.*branch_out .+ slice[2,:]' for slice in eachslice(trunk_out, dims=2)]..., dims=3)
    merged = permutedims(merged, (1,3,2))
    merged = m.downsample(merged)

    return Flux.flatten(merged)
end

@Flux.layer TideModel

##########
# Training
##########

function compute_loss(model, settings::TideSettings, data)
    x_station, x_doodson, y = data
    y_hat = model(x_station, x_doodson)
    return sqrt(Flux.mse(y_hat, y))
end

function train_epoch!(model, settings::TideSettings, train_settings::TrainingSettings, dataloader, opt_state)
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

    lats = get_latitudes(ts)
    lons = get_longitudes(ts)

    # station_ids = 1:length(stations)
    (x_station, x_doodson) = prepare_inputs(settings, lats, lons, times)
    y_hat = model(x_station, x_doodson)
    return reshape(y_hat, length(stations), length(times))
end

function plot_fft(signal, times, label)
    n = length(signal)
    dt = (times[2] - times[1]).value / 3.6e6

    fft = fftshift(FFTW.fft(signal)) * 2 / n
    freqs = fftshift(fftfreq(n, 1/dt))

    fig = plot(freqs, abs.(fft), xlabel="Frequency (1/Hrs)", ylabel="Amplitude", xlims=(0,0.5), label=label)

    return fig
end

function plot_fft!(fig, signal, times, label)
    n = length(signal)
    dt = (times[2] - times[1]).value / 3.6e6
    fft = fftshift(FFTW.fft(signal)) * 2 / n
    freqs = fftshift(fftfreq(n, 1/dt))

    plot!(fig, freqs, abs.(fft), xlabel="Frequency (1/Hrs)", ylabel="Amplitude", xlims=(0,0.5), label=label)
end

function plot_series(model, settings::TideSettings, data_dict::Dict{String, TimeSeries}, series_name; 
    timerange::Union{Vector{DateTime}, Vector{String}, Nothing}=nothing,
    station_names::Union{Vector{String}, Nothing}=nothing,
    show_fft=false, 
    write_series=false, write_format="jld2")

    ts = data_dict["waterlevel"]

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
        p1 = plot(times, h, label="Ground Truth", xlabel="Time", ylabel="Waterlevel", title="Station $(station) RMSE=$(rmse)")
        plot!(p1, times, h_hat, label="Predicted")
        p2 = plot(times, err, label="Residual")

        if show_fft
            p3 = plot_fft(h, times, "Ground Truth FFT")
            plot_fft!(p3, h_hat, times, "Predicted FFT")
            p4 = plot_fft(err, times, "Residual FFT")
            plot(p1,p2,p3,p4,layout=(4,1))
        else
            plot(p1,p2,layout=(2,1))
        end

        # plot(p1,p2,layout=(2,1))
        savefig(joinpath(settings.model_dir, "$(station)_$(series_name).png"))
    end

    if write_series
        fn_pred = joinpath(settings.model_dir, "$(series_name)_tides")
        fn_res = joinpath(settings.model_dir, "$(series_name)_surge")
        station_x = Float64.(get_longitudes(ts))
        station_y = Float64.(get_latitudes(ts))

        if write_format == "netcdf"
            write_to_netcdf(TimeSeries(Float32.(prediction), times, stations, station_x, station_y, "waterlevel", get_source(ts)), fn_pred*".nc")
            write_to_netcdf(TimeSeries(Float32.(errors),     times, stations, station_x, station_y, "surge",      get_source(ts)), fn_res*".nc")
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
                    "waterlevel" => prediction
                )    
            )
            save(fn_res*ext,
                Dict(
                    "station_x_coordinate" => station_x,
                    "station_y_coordinate" => station_y,
                    "station_names" => stations,
                    "times" => times,
                    "waterlevel" => errors
                )    
            )       
        end
    end

end
