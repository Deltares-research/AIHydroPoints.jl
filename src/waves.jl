using Flux
using CUDA
using Statistics
using JLD2
using Dates
using DataFrames

"""
    struct WaveSettings

Inference-time parameters for the wave model.
Training hyperparameters (epochs, learning rate, noise, etc.) are stored separately in `TrainingSettings`.

# Fields

- `model_name`: Name of the model.
    (**Default**: `"MyWaveModel"`)
- `model_dir`: Directory where files generated during the run will be saved.
    (**Default**: `"MyWaveModel"`)
- `use_gpu`: Whether to train/run on GPU.
    (**Default**: `false`)
- `nstations`: Number of wave output stations. Set from training data.
    (**Default**: `nothing`)
- `nwind`: Number of wind input stations. Set from training data.
    (**Default**: `nothing`)
- `nlags`: Number of previous timesteps used as input.
    (**Default**: `16`)
- `n_input_channels`: Number of channels in the first convolutional layer.
    (**Default**: `64`)
- `wind_scale`: Divisor applied to wind stress values before input to model.
    (**Default**: `0.5`)
- `wave_scale`: Divisor applied to wave height values.
    (**Default**: `3.0`)
- `model_pars`: Dict of model architecture parameters.
    (**Default**: `Dict("nchannel" => [64,64,64,1], "activation" => "swish")`)
"""
@kwdef mutable struct WaveSettings <: AbstractModelSettings
    model_name = "MyWaveModel"
    model_dir = "MyWaveModel"
    use_gpu = false
    nstations = nothing   # output (wave) stations — set from data
    nwind = nothing       # input (wind) stations — set from data
    nlags = 16
    n_input_channels = 64
    wind_scale = 0.5
    wave_scale = 3.0
    model_pars = Dict(
        "nchannel" => [64, 64, 64, 1],
        "activation" => "swish",
    )
end

###################
# Input Preparation
###################

"""
    prepare_train_data(data_dict::Dict{String, <:AbstractTimeSeries}, settings::WaveSettings)

Prepare training data for the wave model.

Expected keys in `data_dict`: `"wind_speed"`, `"wind_direction"`, `"wave_height"`.
Wind speed and direction are converted to stress components, scaled, and assembled
into lagged input blocks.  One-hot station encoding is used as the station identifier.
Records containing NaNs are removed.

Returns `(x_station, x_input, y_wave)`.
"""
function prepare_train_data(data_dict::Dict{String, <:AbstractTimeSeries}, settings::WaveSettings)
    u10  = data_dict["wind_speed"]
    udir = data_dict["wind_direction"]
    swh  = data_dict["wave_height"]

    nlags      = settings.nlags
    wind_scale = settings.wind_scale
    wave_scale = settings.wave_scale

    ntimes     = length(get_times(swh))
    itimes     = nlags:ntimes
    ntraining  = length(itimes)

    n_target = length(get_names(swh))
    n_source = length(get_names(u10))

    wind_x, wind_y = _wind_to_stress(get_values(u10), get_values(udir), wind_scale)
    swh_values = Float32.(get_values(swh)) ./ wave_scale

    # One-hot station encoding: (n_target, n_target * ntraining)
    station_arr = collect(1:n_target) * ones(Int, ntraining)'
    x_station = Flux.onehotbatch(station_arr[:], 1:n_target)

    # Wind input blocks: (nlags, n_source*2, n_target*ntraining)
    x_input = zeros(Float32, nlags, n_source * 2, n_target * ntraining)
    for itime in itimes
        x_block = Float32.(vcat(wind_x[:, itime-nlags+1:itime],
                                wind_y[:, itime-nlags+1:itime]))'
        for istation in 1:n_target
            isample = (itime - nlags) * n_target + istation
            x_input[:, :, isample] .= x_block
        end
    end

    # Wave height targets: (1, n_target * ntraining)
    y_wave = reshape(swh_values[:, nlags:end], 1, :)

    # Drop records with NaNs
    valid = [i for i in 1:size(y_wave, 2)
             if !any(isnan, x_input[:, :, i]) &&
                !any(isnan, x_station[:, i])  &&
                !isnan(y_wave[1, i])]
    return x_station[:, valid], x_input[:, :, valid], y_wave[:, valid]
end

# Internal helper: convert (u10, udir) to scaled wind stress components.
function _wind_to_stress(u10_values, udir_values, wind_scale)
    wind_x = Float32.(u10_values .* -sind.(udir_values) ./ wind_scale)
    wind_y = Float32.(u10_values .* -cosd.(udir_values) ./ wind_scale)
    for i in eachindex(wind_x)
        wind_x[i], wind_y[i] = uv_to_stress_xy(wind_x[i], wind_y[i])
    end
    return wind_x, wind_y
end

###############
# Model Builder
###############

"""
    create_wave_model(settings::WaveSettings)

Build a wave model from the hyperparameters in `settings`.

The architecture is: WaveInputLayer → N×Conv(stride=2) → flatten.
`nlags` must equal `2^length(nchannel)`.
"""
function create_wave_model(settings::WaveSettings)
    nstations = settings.nstations
    nlags     = settings.nlags
    npars     = 2 * settings.nwind
    n_ch      = settings.n_input_channels
    nchannel  = settings.model_pars["nchannel"]
    act_name  = get(settings.model_pars, "activation", "swish")
    f_act     = act_name == "relu" ? relu : swish

    @assert nlags == 2^length(nchannel) "nlags must equal 2^length(nchannel)"

    in_ch  = [n_ch; nchannel[1:end-1]]
    out_ch = nchannel
    acts   = [fill(f_act, length(nchannel)-1); [identity]]

    return Chain(
        WaveInputLayer(nstations, nlags, npars, n_ch, f_act),
        [Conv((2,), inc => outc, act, stride=(2,))
         for (inc, outc, act) in zip(in_ch, out_ch, acts)]...,
        Flux.flatten,
    )
end

##########
# Training
##########

function compute_loss(model, settings::WaveSettings, data)
    x_station, x_input, y = data
    y_hat = model((x_station, x_input))
    return sqrt(Flux.mse(y_hat, y))
end

function train_epoch!(model, settings::WaveSettings, train_settings::TrainingSettings, dataloader, opt_state)
    noise_std = Float32(train_settings.input_noise_std)
    acc_loss  = 0.0f0
    for (x_station, x_input, y) in dataloader
        x_noisy = noise_std > 0.0f0 ?
            x_input .+ noise_std .* (x_input isa CuArray ?
                CUDA.randn(Float32, size(x_input)...) :
                randn(Float32, size(x_input))) : x_input
        dloss, grads = Flux.withgradient(model) do m
            Flux.mse(m((x_station, x_noisy)), y)
        end
        Flux.update!(opt_state, model, grads[1])
        acc_loss += dloss
    end
    return acc_loss
end

##############
# Prediction
##############

"""
    predict(model, settings::WaveSettings, data_dict)

Run the wave model on `data_dict` and return predictions as a `TimeSeries`.

Expected keys: `"wind_speed"`, `"wind_direction"`, `"wave_height"`.
The first `nlags-1` time steps are filled with `NaN`.
"""
function predict(model, settings::WaveSettings, data_dict::Dict{String, <:AbstractTimeSeries})
    u10  = data_dict["wind_speed"]
    udir = data_dict["wind_direction"]
    swh  = data_dict["wave_height"]

    nlags      = settings.nlags
    wave_scale = settings.wave_scale
    nstations  = settings.nstations

    times    = get_times(u10)
    ntimes   = length(times)
    itimes   = nlags:ntimes
    ntraining = length(itimes)
    n_source = length(get_names(u10))

    wind_x, wind_y = _wind_to_stress(get_values(u10), get_values(udir), settings.wind_scale)

    x_input = zeros(Float32, nlags, n_source * 2, ntraining)
    for itime in itimes
        x_input[:, :, itime-nlags+1] .= Float32.(
            vcat(wind_x[:, itime-nlags+1:itime],
                 wind_y[:, itime-nlags+1:itime]))'
    end

    y_hat = zeros(Float32, nstations, ntimes)
    y_hat[:, 1:nlags-1] .= NaN
    for istation in 1:nstations
        x_station = Flux.onehotbatch(fill(istation, ntraining), 1:nstations)
        y_hat[istation, nlags:end] .= wave_scale .* model((x_station, x_input))[1, :]
    end

    return TimeSeries(y_hat, times,
                      get_names(swh), get_longitudes(swh), get_latitudes(swh),
                      "wave_height", "wave_model")
end

###############
# Plotting
###############

"""
    plot_series(model, settings::WaveSettings, data_dict, series_name; kwargs...)

Plot predicted vs. target wave heights for each station and optionally write output files.

# Keywords

- `timerange`: Two-element vector of `DateTime` or `String` to restrict the time axis.
- `station_names`: Subset of station names to plot.
- `write_series`: Write predicted and residual series to disk.
    (**Default**: `false`)
- `write_format`: `"jld2"` or `"netcdf"`.
    (**Default**: `"jld2"`)
"""
function plot_series(model, settings::WaveSettings,
                     data_dict::Dict{String, <:AbstractTimeSeries}, series_name;
                     timerange::Union{Vector{DateTime}, Vector{String}, Nothing}=nothing,
                     station_names::Union{Vector{String}, Nothing}=nothing,
                     write_series=false, write_format="jld2")

    u10  = data_dict["wind_speed"]
    udir = data_dict["wind_direction"]
    swh  = data_dict["wave_height"]

    if !isnothing(timerange)
        u10  = select_timespan(u10,  timerange[1], timerange[2])
        udir = select_timespan(udir, timerange[1], timerange[2])
        swh  = select_timespan(swh,  timerange[1], timerange[2])
    end
    if !isnothing(station_names)
        swh = select_locations_by_names(swh, station_names)
    end

    local_dict = Dict("wind_speed" => u10, "wind_direction" => udir, "wave_height" => swh)
    swh_pred = predict(model, settings, local_dict)

    nlags    = settings.nlags
    stations = get_names(swh)
    times    = get_times(swh)[nlags:end]
    y_true   = get_values(swh)[:, nlags:end]
    y_pred   = get_values(swh_pred)[:, nlags:end]
    errors   = y_true .- y_pred
    rmses    = sqrt.(mean(abs2, errors; dims=2))

    for (ind, station) in enumerate(stations)
        h     = y_true[ind, :]
        h_hat = y_pred[ind, :]
        err   = errors[ind, :]
        rmse  = rmses[ind]
        p1 = plot(times, h, label="Target", xlabel="Time",
                  ylabel="Wave height (m)", title="Station $station  RMSE=$(round(rmse, digits=3))")
        plot!(p1, times, h_hat, label="Predicted")
        p2 = plot(times, err, label="Residual")
        plot(p1, p2, layout=(2, 1))
        savefig(joinpath(settings.model_dir, "$(station)_$(series_name).png"))
    end

    if write_series
        _write_wave_series(settings, series_name, times, y_pred, errors, stations, swh, write_format)
    end
end

function _write_wave_series(settings, series_name, times, y_pred, errors, stations, swh, fmt)
    fn_pred = joinpath(settings.model_dir, "$(series_name)_wave_height")
    fn_res  = joinpath(settings.model_dir, "$(series_name)_residual")
    sx = Float64.(get_longitudes(swh))
    sy = Float64.(get_latitudes(swh))
    if fmt == "netcdf"
        write_to_netcdf(TimeSeries(Float32.(y_pred),  times, stations, sx, sy, "wave_height", get_source(swh)), fn_pred*".nc")
        write_to_netcdf(TimeSeries(Float32.(errors),  times, stations, sx, sy, "residual",    get_source(swh)), fn_res*".nc")
    else
        fmt != "jld2" && @warn "Unknown format $fmt, using JLD2."
        save(fn_pred*".jld2", Dict("station_x_coordinate"=>sx, "station_y_coordinate"=>sy,
            "station_names"=>stations, "times"=>times, "wave_height"=>y_pred))
        save(fn_res*".jld2",  Dict("station_x_coordinate"=>sx, "station_y_coordinate"=>sy,
            "station_names"=>stations, "times"=>times, "wave_height"=>errors))
    end
end

###########
# Statistics
###########

"""
    stats_skipnan(y_true::TimeSeries, y_pred::TimeSeries)

Compute per-station statistics (bias, RMSE, MAE, relative bias, scatter index),
skipping NaN values.  Returns a `DataFrame` with one row per station.
"""
function stats_skipnan(y_true::TimeSeries, y_pred::TimeSeries)
    @assert get_times(y_true) == get_times(y_pred) "Times of true and predicted series differ"
    @assert get_names(y_true) == get_names(y_pred) "Station names of true and predicted series differ"

    names        = get_names(y_true)
    y_true_vals  = copy(Float32.(get_values(y_true)))
    y_pred_vals  = Float32.(get_values(y_pred))
    res          = y_pred_vals .- y_true_vals
    count_notnan = sum(.!isnan.(res), dims=2)

    res[isnan.(res)]               .= 0.0f0
    y_true_vals[isnan.(y_true_vals)] .= 0.0f0
    rel_res = res ./ max.(y_true_vals, 0.1f0)

    bias          = sum(res,          dims=2) ./ count_notnan
    rmse          = sqrt.(sum(res.^2, dims=2) ./ count_notnan)
    mae           = sum(abs.(res),    dims=2) ./ count_notnan
    relative_bias = sum(rel_res,      dims=2) ./ count_notnan
    scatter_index = sqrt.(sum(rel_res.^2, dims=2) ./ count_notnan)

    return DataFrame(
        station_name  = names,
        bias          = vec(bias),
        rmse          = vec(rmse),
        mae           = vec(mae),
        relative_bias = vec(relative_bias),
        scatter_index = vec(scatter_index),
        count         = vec(count_notnan),
    )
end

"""
    average_stats(previous_stats, stats::DataFrame, timespan_name::String)

Append a row of station-averaged statistics to `previous_stats` (or create a new
DataFrame if `previous_stats === nothing`).
"""
function average_stats(previous_stats, stats::DataFrame, timespan_name::String)
    row = DataFrame(
        timespan          = timespan_name,
        avg_bias          = mean(stats.bias),
        avg_rmse          = mean(stats.rmse),
        avg_mae           = mean(stats.mae),
        avg_relative_bias = mean(stats.relative_bias),
        avg_scatter_index = mean(stats.scatter_index),
        nstations         = nrow(stats),
    )
    return previous_stats === nothing ? row : vcat(previous_stats, row)
end
